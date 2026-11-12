"""HAM10000: utilities embedded in the standalone Colab notebooks."""
from pathlib import Path
import os, json, hashlib, random, time, platform, shutil, zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

PIPELINE_VERSION = 'colab-2026-11-09-v1'
CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']
POSITIVE = {'mel','bcc','akiec'}

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def safe_extract(archive, destination):
    destination=Path(destination).resolve(); destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if any(not (destination/n).resolve().is_relative_to(destination) for n in z.namelist()):
            raise ValueError('Unsafe ZIP member')
        z.extractall(destination)

def image_index(data):
    index={}
    for p in Path(data).rglob('*.jpg'):
        if p.stem in index: raise ValueError(f'Duplicate image_id on disk: {p.stem}')
        index[p.stem]=p
    return index

def metadata(data):
    df=pd.read_csv(Path(data)/'HAM10000_metadata.csv')
    assert {'image_id','lesion_id','dx'} <= set(df.columns)
    assert not df[['image_id','lesion_id','dx']].isna().any().any()
    assert df.image_id.is_unique and set(df.dx)==set(CLASSES)
    assert len(df)==10015, 'Unexpected dataset version; review before proceeding.'
    assert df.groupby('lesion_id').dx.nunique().max()==1
    return df

def validate_parts(parts, meta):
    for name, frame in parts.items():
        assert frame.image_id.is_unique
        assert set(frame.dx)==set(CLASSES), f'Missing class in {name}'
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        assert set(parts[a].lesion_id).isdisjoint(parts[b].lesion_id)
        assert set(parts[a].image_id).isdisjoint(parts[b].image_id)
    combined=pd.concat(parts.values())[['image_id','lesion_id','dx']].sort_values('image_id').reset_index(drop=True)
    expected=meta[['image_id','lesion_id','dx']].sort_values('image_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(combined,expected)

def split_fingerprint(parts):
    return digest({n:f[['image_id','lesion_id','dx']].sort_values('image_id').to_dict('records') for n,f in parts.items()})

def prepare_splits(data, split_dir, seed=42):
    meta=metadata(data); split_dir=Path(split_dir); split_dir.mkdir(parents=True,exist_ok=True)
    paths={n:split_dir/f'{n}.csv' for n in ['train','valid','test']}
    existing=[p.exists() for p in paths.values()]
    if any(existing) and not all(existing): raise RuntimeError('Incomplete manifests; restore all three partitions.')
    if all(existing):
        parts={n:pd.read_csv(p)[['image_id','lesion_id','dx']] for n,p in paths.items()}
    else:
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=seed).split(meta,groups=meta.lesion_id))
        train=meta.iloc[a]; remaining=meta.iloc[b]
        c,d=next(GroupShuffleSplit(n_splits=1,test_size=.50,random_state=seed).split(remaining,groups=remaining.lesion_id))
        parts={'train':train,'valid':remaining.iloc[c],'test':remaining.iloc[d]}
        parts={n:f[['image_id','lesion_id','dx']].copy() for n,f in parts.items()}
    validate_parts(parts,meta)
    for n,f in parts.items(): f.to_csv(paths[n],index=False)
    write_json(split_dir/'split_info.json',{'seed':seed,'fingerprint':split_fingerprint(parts),
        'counts':{n:{'images':len(f),'lesions':f.lesion_id.nunique()} for n,f in parts.items()}})
    return parts

def audit_images(data, parts, audit_dir):
    audit_dir=Path(audit_dir); audit_dir.mkdir(parents=True,exist_ok=True)
    index=image_index(data); rows=[]; failures=[]
    for split, frame in parts.items():
        for row in tqdm(frame.itertuples(),total=len(frame),desc=f'Audit {split}'):
            p=index.get(row.image_id)
            try:
                if p is None: raise FileNotFoundError(row.image_id)
                with Image.open(p) as im:
                    width,height=im.size; im.verify()
                sha=hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append({'image_id':row.image_id,'split':split,'sha256':sha,'width':width,'height':height})
            except Exception as e: failures.append({'image_id':row.image_id,'error':str(e)})
    table=pd.DataFrame(rows)
    cross=table.groupby('sha256')['split'].nunique() if len(table) else pd.Series(dtype=int)
    bad_hashes=cross[cross>1].index
    duplicates=table[table.duplicated('sha256',keep=False)] if len(table) else table
    duplicates.to_csv(audit_dir/'duplicates.csv',index=False)
    table.to_csv(audit_dir/'image_audit.csv',index=False)
    result={'content_hash':digest(table[['image_id','sha256']].sort_values('image_id').to_dict('records')),'split_fingerprint':split_fingerprint(parts),'failures':failures,
            'cross_split_duplicate_hashes':list(bad_hashes),'images_checked':len(table)}
    write_json(audit_dir/'audit.json',result)
    assert not failures, f'{len(failures)} corrupt/missing images; inspect audit.json'
    assert not len(bad_hashes), 'Exact duplicates cross partitions; resolve and document before training.'
    return table

def load_parts(data, split_dir, audit_dir, task, smoke=False):
    meta=metadata(data)
    parts={n:pd.read_csv(Path(split_dir)/f'{n}.csv')[['image_id','lesion_id','dx']] for n in ['train','valid','test']}
    validate_parts(parts,meta); fp=split_fingerprint(parts)
    audit=json.loads((Path(audit_dir)/'audit.json').read_text(encoding='utf-8'))
    assert audit['split_fingerprint']==fp and audit['images_checked']==len(meta)
    assert not audit['failures'] and not audit['cross_split_duplicate_hashes']
    index=image_index(data)
    for n,f in parts.items():
        f['image_path']=f.image_id.map(index)
        assert not f.image_path.isna().any()
        f['target']=f.dx.map({c:i for i,c in enumerate(CLASSES)}) if task=='multiclass' else f.dx.isin(POSITIVE).astype(int)
        if smoke and n!='test':
            parts[n]=pd.concat([g.sample(min(4,len(g)),random_state=42) for _,g in f.groupby('target')]).reset_index(drop=True)
    return parts,fp

class SkinDataset(Dataset):
    def __init__(self,frame,augment):
        self.frame=frame.reset_index(drop=True)
        ops=[transforms.Resize((224,224))]
        if augment: ops += [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(20)]
        self.transform=transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        with Image.open(r.image_path) as image: tensor=self.transform(image.convert('RGB'))
        return tensor,int(r.target)

class Baseline(nn.Module):
    def __init__(self,k):
        super().__init__(); layers=[]; channels=3
        for out in [32,64,128]:
            layers.extend([nn.Conv2d(channels,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(),nn.MaxPool2d(2)])
            channels=out
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.3),nn.Linear(128,k))
    def forward(self,x): return self.net(x)

def make_model(name,k,pretrained=True):
    if name=='baseline': return Baseline(k)
    if name=='vgg16':
        m=models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        m.avgpool=nn.AdaptiveAvgPool2d(1); m.classifier=nn.Sequential(nn.Dropout(.3),nn.Linear(512,k))
    elif name=='resnet50':
        m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None); m.fc=nn.Linear(m.fc.in_features,k)
    elif name=='efficientnet_b0':
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,k)
    else: raise ValueError(name)
    return m

def set_stage(model,name,stage):
    for p in model.parameters(): p.requires_grad=(stage=='all')
    if stage=='all': return
    head=model.fc if name=='resnet50' else model.classifier
    for p in head.parameters(): p.requires_grad=True
    if stage=='finetune':
        blocks=model.layer4 if name=='resnet50' else model.features[24:] if name=='vgg16' else model.features[-2:]
        for p in blocks.parameters(): p.requires_grad=True

def loader(frame,augment,batch,seed,strategy='none'):
    generator=torch.Generator().manual_seed(seed)
    sampler=None
    if strategy=='sampler':
        counts=frame.target.value_counts(); weights=frame.target.map(1/counts).to_numpy(copy=True)
        sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(frame),replacement=True,generator=generator)
    # 0 workers keeps notebook-defined datasets compatible with Windows and Colab.
    return DataLoader(SkinDataset(frame,augment),batch_size=batch,shuffle=augment and sampler is None,
                      sampler=sampler,num_workers=0,pin_memory=torch.cuda.is_available(),generator=generator)

def predict(model,frame,batch,device):
    model.eval(); ys=[]; probabilities=[]
    with torch.inference_mode():
        for x,y in loader(frame,False,batch,0):
            probabilities.append(model(x.to(device,non_blocking=True)).softmax(1).cpu().numpy()); ys.extend(y.tolist())
    return np.asarray(ys),np.concatenate(probabilities)

def metrics(y,prob,names):
    pred=prob.argmax(1); labels=list(range(len(names)))
    auc=None
    try: auc=float(roc_auc_score(y,prob[:,1] if len(names)==2 else prob,**({} if len(names)==2 else {'multi_class':'ovr','average':'macro','labels':labels})))
    except ValueError: pass
    return {'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'f1_macro':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
            'auc_roc':auc,'report':classification_report(y,pred,labels=labels,target_names=names,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'class_order':names, 'sensitivity':{name:float((pred[y==i]==i).mean()) if (y==i).any() else None for i,name in enumerate(names)}, 'false_negative_rate':{name:float((pred[y==i]!=i).mean()) if (y==i).any() else None for i,name in enumerate(names)}}

def save_evaluation(model,frame,batch,device,names,destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    y,prob=predict(model,frame,batch,device); result=metrics(y,prob,names)
    write_json(destination/'metrics.json',result)
    predictions=frame[['image_id','lesion_id','dx']].copy(); predictions['target']=y; predictions['prediction']=prob.argmax(1)
    for i,name in enumerate(names): predictions[f'prob_{name}']=prob[:,i]
    predictions.to_csv(destination/'predictions.csv',index=False)
    pd.DataFrame(result['report']).T.to_csv(destination/'classification_report.csv')
    cm=np.asarray(result['confusion_matrix']); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,xlabel='Predicted',ylabel='True')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(destination/'confusion_matrix.png'); plt.close(fig)
    fig,ax=plt.subplots()
    for i,name in enumerate(names):
        if len(names)==2 and i==0: continue
        binary=(y==i).astype(int)
        if len(np.unique(binary))==2:
            fpr,tpr,_=roc_curve(binary,prob[:,i]); ax.plot(fpr,tpr,label=name)
    ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='False positive rate',ylabel='True positive rate'); ax.legend()
    fig.tight_layout(); fig.savefig(destination/'roc.png'); plt.close(fig)
    return result

def checkpoint_save(path,value):
    path=Path(path); temp=path.with_suffix('.tmp'); torch.save(value,temp); os.replace(temp,path)

def train_run(name,k,parts,fp,run_dir,config,device):
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    identity={'pipeline':PIPELINE_VERSION,'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,'model':name,'classes':k,'split':fp,'config':config}; signature=digest(identity)
    manifest=run_dir/'config.json'
    if manifest.exists(): assert json.loads(manifest.read_text())['signature']==signature, 'Config changed: choose a new EXPERIMENT_ID.'
    write_json(manifest,{'signature':signature,**identity,'python':platform.python_version(),'torch':torch.__version__,
                         'torchvision':__import__('torchvision').__version__,'device':str(device),
                         'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None})
    seed_all(config['seed']); model=make_model(name,k,pretrained=not (run_dir/'last.pt').exists()).to(device)
    plan=[(stage['name'],stage['lr']) for stage in config['stages'] for _ in range(stage['epochs'])]
    history=[]; best=-1.; stale=0; start=0; prior_stage=None; saved=None
    last=run_dir/'last.pt'; best_path=run_dir/'best.pt'
    if last.exists():
        # Only load checkpoints created by this project (torch.load uses pickle).
        saved=torch.load(last,map_location=device,weights_only=False)
        assert saved['signature']==signature
        model.load_state_dict(saved['model']); history=saved['history']; best=saved['best']; stale=saved['stale']; start=saved['epoch']+1; prior_stage=saved['stage']
        if saved.get('finished'): start=len(plan)
        print(f'Resuming {name} at epoch {start+1}',flush=True)
    counts=np.bincount(parts['train'].target,minlength=k)
    assert np.all(counts>0)
    class_weights=torch.tensor(len(parts['train'])/(k*counts),device=device,dtype=torch.float32) if config['strategy']=='weighted' else None
    criterion=nn.CrossEntropyLoss(weight=class_weights); optimizer=None; scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    for epoch in range(start,len(plan)):
        stage,lr=plan[epoch]; seed_all(config['seed']+epoch)
        if optimizer is None or stage!=prior_stage:
            if stage!=prior_stage and prior_stage is not None: model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model'])
            set_stage(model,name,stage)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=config['weight_decay'])
            if saved is not None and stage==saved['stage']:
                optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
            else: stale=0
        prior_stage=stage; model.train()
        if stage!='all':
            model.eval()
            (model.fc if name=='resnet50' else model.classifier).train()
            if stage=='finetune':
                blocks=model.layer4 if name=='resnet50' else model.features[24:] if name=='vgg16' else model.features[-2:]
                blocks.train()
        # Keep running statistics fixed in frozen backbone blocks.
        for mod in model.modules():
            if isinstance(mod,nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()): mod.eval()
        started=time.perf_counter(); total=0.; dl=loader(parts['train'],True,config['batch_size'],config['seed']+epoch,config['strategy'])
        for x,y in tqdm(dl,desc=f'{name} {stage} epoch {epoch+1}/{len(plan)}'):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=criterion(model(x),y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; inspect inputs/learning rate.')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total+=float(loss.detach())*len(y)
        yv,prob=predict(model,parts['valid'],config['batch_size'],device)
        score=metrics(yv,prob,['benign','malignant_or_premalignant'] if k==2 else CLASSES)
        improved=score['balanced_accuracy']>best
        if improved: best=score['balanced_accuracy']; stale=0
        else: stale+=1
        history.append({'epoch':epoch+1,'stage':stage,'train_loss':total/len(parts['train']),
                        'valid_loss':float(-np.log(np.clip(prob[np.arange(len(yv)),yv],1e-8,1)).mean()),
                        'valid_balanced_accuracy':score['balanced_accuracy'],'valid_f1_macro':score['f1_macro'],
                        'seconds':time.perf_counter()-started})
        stop=stage!='head' and stale>=config['patience']
        state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch,'stage':stage,'best':best,'stale':stale,'history':history,'finished':stop or epoch==len(plan)-1}
        if improved: checkpoint_save(best_path,{'signature':signature,'model':model.state_dict(),'epoch':epoch,'score':score})
        checkpoint_save(last,state); pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        print(history[-1],flush=True)
        if stop: print('Early stopping on validation.',flush=True); break
    best_state=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(best_state['model'])
    names=['benign','malignant_or_premalignant'] if k==2 else CLASSES
    result=save_evaluation(model,parts['valid'],config['batch_size'],device,names,run_dir/'validation')
    summary={'model':name,'run_dir':str(run_dir),'signature':signature,'split':fp,'mode':config['mode'],
             'balanced_accuracy':result['balanced_accuracy'],'f1_macro':result['f1_macro'],'best_epoch':best_state['epoch']+1}
    write_json(run_dir/'summary.json',summary)
    hist=pd.DataFrame(history); fig,axes=plt.subplots(1,2,figsize=(10,4))
    hist.plot(x='epoch',y=['train_loss','valid_loss'],ax=axes[0]); hist.plot(x='epoch',y=['valid_balanced_accuracy','valid_f1_macro'],ax=axes[1])
    fig.tight_layout(); fig.savefig(run_dir/'learning_curves.png'); plt.close(fig)
    del model,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return summary

def select_run(rows,path):
    assert rows and all(r['mode']==rows[0]['mode'] and r['split']==rows[0]['split'] for r in rows)
    chosen=sorted(rows,key=lambda r:(-r['balanced_accuracy'],-r['f1_macro'],r['model'],r['run_dir']))[0]
    write_json(path,{'criterion':'validation balanced accuracy; tie: F1 macro then stable model/run name','selected':chosen,'candidates':rows})
    return chosen

def final_test(selected,parts,fp,destination,device,batch):
    assert selected['mode']=='full' and selected['split']==fp
    destination=Path(destination)
    manifest=json.loads((Path(selected['run_dir'])/'config.json').read_text())
    assert manifest['config']['mode']=='full' and manifest['signature']==selected['signature']
    if (destination/'lock.json').exists():
        assert json.loads((destination/'lock.json').read_text())['signature']==selected['signature']
    if (destination/'metrics.json').exists():
        assert json.loads((destination/'lock.json').read_text())['signature']==selected['signature']
        print('Existing final evaluation preserved:',destination); return
    write_json(destination/'lock.json',{'signature':selected['signature'],'split':fp})
    model=make_model(selected['model'],len(CLASSES) if parts['test'].target.max()>1 else 2,pretrained=False).to(device)
    state=torch.load(Path(selected['run_dir'])/'best.pt',map_location=device,weights_only=False)
    assert state['signature']==selected['signature']; model.load_state_dict(state['model'])
    names=CLASSES if parts['test'].target.max()>1 else ['benign','malignant_or_premalignant']
    save_evaluation(model,parts['test'],batch,device,names,destination)
    del model
    if device.type=='cuda': torch.cuda.empty_cache()
