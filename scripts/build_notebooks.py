from pathlib import Path
import nbformat, textwrap
BASE=Path(__file__).resolve().parents[1]
HELPERS=(BASE/'src'/'ham10000.py').read_text(encoding='utf-8-sig')
SETUP='''# ConfiguraciÃ³n comÃºn: usa los mismos valores en los cinco notebooks.
import os, sys, subprocess, importlib.util
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
MODE = 'full'  # 'smoke' permite comprobaciÃ³n breve local; resultados separados.
EXPERIMENT_ID = 'colab_v1'  # Cambia este nombre si modificas hiperparÃ¡metros.
IS_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PERSIST = Path('/content/drive/MyDrive/HAM10000_PROJECT')
    DATA_ROOT = Path('/content/ham10000')  # ImÃ¡genes locales: evita leer miles de JPG desde Drive.
else:
    PROJECT = Path.cwd()
    if not (PROJECT/'notebooks').exists() and (PROJECT.parent/'notebooks').exists(): PROJECT=PROJECT.parent
    PERSIST = PROJECT/'colab_artifacts'
    DATA_ROOT = PROJECT/'data'/'ham10000'
assert MODE in {'full','smoke'}
SPLIT_DIR=PERSIST/'splits'; AUDIT_DIR=PERSIST/'audit'
RUN_ROOT=PERSIST/'runs'/MODE/EXPERIMENT_ID
CACHE_ZIP=PERSIST/'dataset_cache'/'skin-cancer-mnist-ham10000.zip'
for p in [PERSIST,RUN_ROOT]: p.mkdir(parents=True,exist_ok=True)
# Colab ya incluye torch/torchvision. No se reemplazan sus versiones CUDA.
for package,module in [('pandas','pandas'),('numpy','numpy'),('scikit-learn','sklearn'),('matplotlib','matplotlib'),('Pillow','PIL'),('tqdm','tqdm')]:
    if importlib.util.find_spec(module) is None: subprocess.check_call([sys.executable,'-m','pip','install',package])
import torch, torchvision
torch.set_num_threads(2)  # Evita sobrecarga de hilos en transformaciones pequeñas.
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('PyTorch',torch.__version__,'torchvision',torchvision.__version__,'dispositivo',DEVICE)
print('Artefactos persistentes:',PERSIST,'; modo:',MODE)
'''
RESTORE='''# Repite esta celda tras reconectar a otro runtime de Colab.
if not (DATA_ROOT/'HAM10000_metadata.csv').exists() or len(list(DATA_ROOT.rglob('*.jpg')))!=10015:
    if not CACHE_ZIP.exists(): raise FileNotFoundError('Ejecuta primero 00_descarga_datos o copia el ZIP de Kaggle a dataset_cache en Drive.')
    safe_extract(CACHE_ZIP,DATA_ROOT)
print('ImÃ¡genes locales:',len(image_index(DATA_ROOT)))
'''
GPU='''if MODE=='full' and DEVICE.type!='cuda':
    raise RuntimeError('Entrenamiento completo requiere GPU: Colab > Entorno de ejecuciÃ³n > Cambiar tipo de entorno > GPU. Usa MODE=smoke para comprobar el flujo en CPU.')
if DEVICE.type=='cuda': print(torch.cuda.get_device_name(0))
'''
INTRO='''Abre este archivo en Google Colab. En entrenamiento selecciona **Entorno de ejecuciÃ³n â†’ Cambiar tipo de entorno de ejecuciÃ³n â†’ GPU**. Ejecuta en orden 00, 01, 02, 03, 04 con el mismo `EXPERIMENT_ID`.

Los helpers aparecen completos en una celda para que cada notebook funcione sin clonar un repositorio. Su fuente comÃºn es `src/ham10000.py`. Las imÃ¡genes se extraen al disco local del runtime; particiones, auditorÃ­a, checkpoints y resultados se guardan en Drive. No guardes tokens en celdas ni en salidas.
'''
def md(s):return nbformat.v4.new_markdown_cell(s)
def code(s):return nbformat.v4.new_code_cell(textwrap.dedent(s).strip())
def save(name,title,cells):
    badge=f'[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/notebooks/{name})'
    nb=nbformat.v4.new_notebook(cells=[md('# '+title+'\n\n'+badge+'\n\n'+INTRO),code(SETUP),md('## Funciones reproducibles\n\nEsta celda define datos, modelos, entrenamiento y evaluaciÃ³n; ejecÃºtala sin cambios.'),code(HELPERS)]+cells)
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU','colab':{'provenance':[]}}
    nbformat.write(nb,BASE/'notebooks'/name)

save('00_descarga_datos.ipynb','00 â€” ObtenciÃ³n y almacenamiento de HAM10000',[
 md('## Descarga o restauraciÃ³n\n\nLa copia de Kaggle usada declara CC-BY-NC-SA-4.0. Conserva la atribuciÃ³n a Tschandl, Rosendahl y Kittler (2018), DOI 10.1038/sdata.2018.161. Configura `KAGGLE_API_TOKEN` en los **Secretos de Colab** con acceso al notebook, o introdÃºcelo en la entrada oculta. El ZIP opcional en Drive permite restaurar el dataset despuÃ©s de una desconexiÃ³n. Comprueba el espacio disponible; el ZIP ocupa unos 5,2 GB y se necesita espacio adicional para extraerlo.'),
 code('''CACHE_DATASET_ZIP=True
if (DATA_ROOT/'HAM10000_metadata.csv').exists() and len(image_index(DATA_ROOT))==10015:
    print('Dataset completo presente; se omite descarga.')
elif CACHE_ZIP.exists():
    safe_extract(CACHE_ZIP,DATA_ROOT)
else:
    import getpass
    if importlib.util.find_spec('kaggle') is None:
        subprocess.check_call([sys.executable,'-m','pip','install','kaggle>=1.7'])
    token=os.environ.get('KAGGLE_API_TOKEN')
    if IS_COLAB and not token:
        try:
            from google.colab import userdata
            token=userdata.get('KAGGLE_API_TOKEN')
        except Exception:
            pass
    # Local: se admite el access_token ya configurado en data/.kaggle.
    cfg=(Path('/content/.kaggle') if IS_COLAB else DATA_ROOT.parent/'.kaggle')
    cfg.mkdir(parents=True,exist_ok=True); os.environ['KAGGLE_CONFIG_DIR']=str(cfg)
    if not token and not any((cfg/n).exists() for n in ['access_token','kaggle.json','credentials.json']):
        token=getpass.getpass('Kaggle API token (entrada oculta): ')
    if token: os.environ['KAGGLE_API_TOKEN']=token
    archive=DATA_ROOT/'skin-cancer-mnist-ham10000.zip'; DATA_ROOT.mkdir(parents=True,exist_ok=True)
    try:
        subprocess.run([sys.executable,'-m','kaggle','datasets','download','-d','kmader/skin-cancer-mnist-ham10000','-p',str(DATA_ROOT)],check=True)
        if CACHE_DATASET_ZIP:
            CACHE_ZIP.parent.mkdir(parents=True,exist_ok=True)
            pending=CACHE_ZIP.with_suffix('.partial'); shutil.copy2(archive,pending); os.replace(pending,CACHE_ZIP)
        safe_extract(archive,DATA_ROOT)
        archive.unlink()
    finally:
        os.environ.pop('KAGGLE_API_TOKEN',None); token=None
meta=metadata(DATA_ROOT); index=image_index(DATA_ROOT)
assert set(meta.image_id)<=set(index)
print('Verificado:',len(meta),'filas y',len(index),'imÃ¡genes; cachÃ© Drive:',CACHE_ZIP.exists())''')])

save('01_exploracion_y_preprocesado.ipynb','01 â€” ExploraciÃ³n, calidad y particiones por lesiÃ³n',[
 code(RESTORE),
 md('## Metadatos y exploraciÃ³n\n\nSe conserva el 70/15/15 agrupado de la propuesta. Si existen manifiestos en Drive se validan y reutilizan, sin regenerarlos. Los manifiestos contienen IDs y diagnÃ³sticos; las rutas se resuelven en cada runtime.'),
 code('''meta=metadata(DATA_ROOT)
display(meta.isna().sum().to_frame('ausentes'))
display(meta.dx.value_counts().to_frame('imÃ¡genes'))
fig,axes=plt.subplots(1,2,figsize=(12,4))
meta.dx.value_counts().plot.bar(ax=axes[0],title='DiagnÃ³stico: desbalance de clases')
meta.groupby('lesion_id').size().value_counts().sort_index().plot.bar(ax=axes[1],title='FotografÃ­as por lesiÃ³n')
fig.tight_layout(); fig.savefig(PERSIST/'exploration.png'); plt.show()
index=image_index(DATA_ROOT)
fig,axes=plt.subplots(7,2,figsize=(7,18))
for row,diagnosis in enumerate(CLASSES):
    for col,r in enumerate(meta[meta.dx==diagnosis].sample(2,random_state=42).itertuples()):
        with Image.open(index[r.image_id]) as im: axes[row,col].imshow(im.copy())
        axes[row,col].set_title(diagnosis+' / '+r.image_id); axes[row,col].axis('off')
fig.tight_layout(); fig.savefig(PERSIST/'image_examples.png'); plt.show()'''),
 code('''parts=prepare_splits(DATA_ROOT,SPLIT_DIR,seed=42)
display(pd.DataFrame({n:{'imÃ¡genes':len(f),'lesiones':f.lesion_id.nunique()} for n,f in parts.items()}).T)
display(pd.DataFrame({n:f.dx.value_counts() for n,f in parts.items()}).fillna(0))
print('Fingerprint:',split_fingerprint(parts))'''),
 md('## AuditorÃ­a de imÃ¡genes\n\nVerifica todos los JPG, dimensiones y duplicados exactos SHA-256. Una imagen corrupta o un duplicado exacto entre particiones bloquea el entrenamiento para que se resuelva y documente la causa. No se mueve informaciÃ³n entre particiones automÃ¡ticamente.'),
 code('''audit=audit_images(DATA_ROOT,parts,AUDIT_DIR)
display(audit[['width','height']].describe())
print('AuditorÃ­a terminada. Revisa duplicates.csv incluso si las copias estÃ¡n dentro de una sola particiÃ³n.')''')])

BASE_CONFIG='''config={'mode':MODE,'seed':42,'batch_size':BATCH_SIZE,'weight_decay':1e-4,'patience':PATIENCE,
        'strategy':STRATEGY,'stages':STAGES,'audit_hash':json.loads((AUDIT_DIR/'audit.json').read_text())['content_hash']}
'''
save('02_cnn_linea_base.ipynb','02 â€” CNN binaria y tratamiento del desbalance',[
 code(RESTORE),code(GPU),
 md('## Protocolo\n\nClase positiva: melanoma (`mel`), carcinoma basocelular (`bcc`) y lesiones `akiec` malignas o premalignas. ComparaciÃ³n: sin correcciÃ³n, pÃ©rdida ponderada y muestreo balanceado. La ponderaciÃ³n y el muestreo no se combinan. Se selecciona por balanced accuracy de validaciÃ³n; el test queda reservado. Semillas, configuraciones y versiones se registran. Los checkpoints se guardan despuÃ©s de cada Ã©poca; una reconexiÃ³n puede requerir repetir la Ã©poca interrumpida.'),
 code('''BATCH_SIZE=32
PATIENCE=5
EPOCHS=20 if MODE=='full' else 1
parts,fp=load_parts(DATA_ROOT,SPLIT_DIR,AUDIT_DIR,'binary',smoke=MODE=='smoke')
STAGES=[{'name':'all','epochs':EPOCHS,'lr':1e-3}]
rows=[]
for STRATEGY in ['none','weighted','sampler']:
    config={'mode':MODE,'seed':42,'batch_size':BATCH_SIZE,'weight_decay':1e-4,'patience':PATIENCE,'strategy':STRATEGY,'stages':STAGES,'audit_hash':json.loads((AUDIT_DIR/'audit.json').read_text())['content_hash']}
    rows.append(train_run('baseline',2,parts,fp,RUN_ROOT/'baseline'/STRATEGY,config,DEVICE))
chosen=select_run(rows,RUN_ROOT/'baseline_selection.json')
display(pd.DataFrame(rows)[['run_dir','balanced_accuracy','f1_macro','best_epoch']])
print('Seleccionado por validaciÃ³n:',chosen['run_dir'])'''),
 md('Si cambias batch size, Ã©pocas, estrategia o LR, cambia `EXPERIMENT_ID` en todos los notebooks. Un directorio existente con una configuraciÃ³n distinta se rechaza. Las curvas, reportes, AUC, predicciones y checkpoints estÃ¡n bajo el directorio de cada corrida en Drive.')])

save('03_transfer_learning_binario.ipynb','03 â€” VGG16, ResNet50 y EfficientNet-B0',[
 code(RESTORE),code(GPU),
 md('## ComparaciÃ³n comÃºn\n\nPesos ImageNet, imÃ¡genes 224Ã—224 y normalizaciÃ³n ImageNet. VGG16 usa average pooling global y una cabeza pequeÃ±a para reducir memoria. Se entrena la cabeza y luego el Ãºltimo bloque del backbone con LR menor; el resto permanece congelado, incluidas sus estadÃ­sticas BatchNorm. Todas las arquitecturas usan las mismas particiones, semillas, batch, estrategia y presupuesto de Ã©pocas. El ganador se decide mediante validaciÃ³n y se transfiere al notebook 04. No se consulta test.'),
 code('''BATCH_SIZE=16  # comÃºn para las tres arquitecturas; baja este valor si la GPU agota memoria y usa otro EXPERIMENT_ID.
PATIENCE=5
STRATEGY='weighted'
STAGES=([{'name':'head','epochs':5,'lr':1e-3},{'name':'finetune','epochs':15,'lr':1e-4}] if MODE=='full'
        else [{'name':'head','epochs':1,'lr':1e-3},{'name':'finetune','epochs':1,'lr':1e-4}])
parts,fp=load_parts(DATA_ROOT,SPLIT_DIR,AUDIT_DIR,'binary',smoke=MODE=='smoke')
rows=[]
for architecture in ['vgg16','resnet50','efficientnet_b0']:
    config={'mode':MODE,'seed':42,'batch_size':BATCH_SIZE,'weight_decay':1e-4,'patience':PATIENCE,'strategy':STRATEGY,'stages':STAGES,'audit_hash':json.loads((AUDIT_DIR/'audit.json').read_text())['content_hash']}
    rows.append(train_run(architecture,2,parts,fp,RUN_ROOT/'transfer'/architecture,config,DEVICE))
chosen=select_run(rows,RUN_ROOT/'transfer_selection.json')
display(pd.DataFrame(rows)[['model','balanced_accuracy','f1_macro','best_epoch']])
print('Arquitectura para multiclase:',chosen['model'])''')])

save('04_clasificacion_multiclase.ipynb','04 â€” Siete clases y evaluaciÃ³n final reservada',[
 code(RESTORE),code(GPU),
 md('## Ajuste fino progresivo\n\nSe reutiliza la arquitectura ganadora del notebook 03, con pesos ImageNet nuevos y una cabeza de siete clases. Las etapas son cabeza y Ãºltimo bloque; la pÃ©rdida se pondera con frecuencias de entrenamiento. La selecciÃ³n de Ã©poca usa validaciÃ³n. El orden de clases se guarda en las mÃ©tricas.'),
 code('''selection_path=RUN_ROOT/'transfer_selection.json'
if not selection_path.exists(): raise FileNotFoundError('Completa primero el notebook 03.')
selected_binary=json.loads(selection_path.read_text())['selected']
assert selected_binary['mode']==MODE
architecture=selected_binary['model']
BATCH_SIZE=16
PATIENCE=5
STAGES=([{'name':'head','epochs':5,'lr':1e-3},{'name':'finetune','epochs':20,'lr':1e-4}] if MODE=='full'
        else [{'name':'head','epochs':1,'lr':1e-3},{'name':'finetune','epochs':1,'lr':1e-4}])
parts_multi,fp=load_parts(DATA_ROOT,SPLIT_DIR,AUDIT_DIR,'multiclass',smoke=MODE=='smoke')
assert selected_binary['split']==fp
config={'mode':MODE,'seed':42,'batch_size':BATCH_SIZE,'weight_decay':1e-4,'patience':PATIENCE,'strategy':'weighted','stages':STAGES,'audit_hash':json.loads((AUDIT_DIR/'audit.json').read_text())['content_hash']}
multiclass=train_run(architecture,7,parts_multi,fp,RUN_ROOT/'multiclass'/architecture,config,DEVICE)
write_json(RUN_ROOT/'multiclass_selection.json',{'selected':multiclass})
print(multiclass)'''),
 md('## EvaluaciÃ³n final\n\nEjecuta esta celda con `RUN_FINAL_TEST=True` solamente despuÃ©s de completar y revisar los experimentos completos y fijar las configuraciones. Se registra el protocolo y se evalÃºan una vez: la baseline seleccionada, el mejor transfer binario y el modelo multiclase. No se eligen hiperparÃ¡metros a partir del test. Los resultados existentes se conservan; cualquier cambio de selecciÃ³n queda bloqueado por el protocolo. Un modo smoke nunca puede evaluar test.'),
 code('''RUN_FINAL_TEST=False
if RUN_FINAL_TEST:
    assert MODE=='full', 'No se permite test en smoke.'
    baseline=json.loads((RUN_ROOT/'baseline_selection.json').read_text())['selected']
    binary=json.loads((RUN_ROOT/'transfer_selection.json').read_text())['selected']
    multi=json.loads((RUN_ROOT/'multiclass_selection.json').read_text())['selected']
    protocol={'split':fp,'baseline':baseline['signature'],'binary':binary['signature'],'multiclass':multi['signature']}
    lock=RUN_ROOT/'FINAL_PROTOCOL.json'
    if lock.exists(): assert json.loads(lock.read_text())==protocol, 'La selecciÃ³n cambiÃ³ despuÃ©s de fijar el protocolo.'
    else: write_json(lock,protocol)
    parts_binary,fp_binary=load_parts(DATA_ROOT,SPLIT_DIR,AUDIT_DIR,'binary')
    assert fp_binary==fp
    final_test(baseline,parts_binary,fp,RUN_ROOT/'final_test'/'baseline',DEVICE,BATCH_SIZE)
    final_test(binary,parts_binary,fp,RUN_ROOT/'final_test'/'transfer_binary',DEVICE,BATCH_SIZE)
    final_test(multi,parts_multi,fp,RUN_ROOT/'final_test'/'multiclass',DEVICE,BATCH_SIZE)
else:
    print('EvaluaciÃ³n final desactivada. Revisa validaciÃ³n y fija el protocolo antes de activarla.')''')])
print('Generated 5 standalone notebooks')
