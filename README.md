# Clasificación de lesiones cutáneas con HAM10000

Proyecto académico de **Fundamentos de Deep Learning** de la Universidad de Antioquia. Se propone comparar redes neuronales convolucionales para clasificar imágenes dermatoscópicas del dataset HAM10000 utilizando PyTorch.

**Autor:** Milton Alejandro Cuervo Ramirez  
**Programa:** Ingeniería de Sistemas  
**Profesores:** Raúl Ramos y Julián Arias  
**Fecha:** Septiembre de 2026

## Abrir y ejecutar en Google Colab

Abre los notebooks en este orden. Los modelos completos requieren seleccionar GPU en Colab; Drive conserva los checkpoints y resultados. El token de Kaggle se configura mediante Secretos de Colab.

| Paso | Notebook | Ejecutar |
|---|---|---|
| 00 | [00_descarga_datos](notebooks/00_descarga_datos.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/notebooks/00_descarga_datos.ipynb) |
| 01 | [01_exploracion_y_preprocesado](notebooks/01_exploracion_y_preprocesado.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/notebooks/01_exploracion_y_preprocesado.ipynb) |
| 02 | [02_cnn_linea_base](notebooks/02_cnn_linea_base.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/notebooks/02_cnn_linea_base.ipynb) |
| 03 | [03_transfer_learning_binario](notebooks/03_transfer_learning_binario.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/notebooks/03_transfer_learning_binario.ipynb) |
| 04 | [04_clasificacion_multiclase](notebooks/04_clasificacion_multiclase.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/notebooks/04_clasificacion_multiclase.ipynb) |

Consulta [la guía de Colab](COLAB_GUIDE.md). Los notebooks son autónomos; no requieren subir el código a Drive. El notebook 00 obtiene los datos y el 01 genera o reutiliza las particiones. La evaluación de test permanece desactivada hasta fijar el protocolo.

## Estado actual

La propuesta inicial y `ENTREGA1.PDF` se conservan. La Entrega 2 está en implementación: el dataset se descargó, se generaron particiones agrupadas por lesión y se ejecutó una prueba corta de la CNN en CPU. La corrida completa de baseline se interrumpió por su alto tiempo en CPU. Las corridas cortas de una época no constituyen entrenamiento completo ni resultados finales.

## ¿Qué se busca?

El objetivo es estudiar cómo distintas arquitecturas reconocen patrones visuales en lesiones cutáneas y comparar su desempeño bajo condiciones de evaluación comunes.

Además del desempeño general, se analizarán los errores por categoría y el efecto del desbalance de clases: algunas lesiones tienen muchos más ejemplos que otras.

## Dataset

Se utilizará **HAM10000**, un conjunto de 10.015 imágenes dermatoscópicas distribuidas en siete categorías:

| Código | Categoría | Imágenes |
|---|---|---:|
| `nv` | Nevos melanocíticos | 6.705 |
| `mel` | Melanoma | 1.113 |
| `bkl` | Lesiones benignas similares a queratosis | 1.099 |
| `bcc` | Carcinoma basocelular | 514 |
| `akiec` | Queratosis actínicas y carcinoma intraepitelial / enfermedad de Bowen | 327 |
| `vasc` | Lesiones vasculares | 142 |
| `df` | Dermatofibroma | 115 |

Las imágenes están acompañadas de metadatos que incluyen identificadores de imagen y lesión, diagnóstico, método de confirmación, edad, sexo y localización corporal.

Una lesión puede tener varias fotografías. Por esta razón, se utilizará `lesion_id` para mantener sus imágenes dentro de una misma partición y evitar que aparezcan simultáneamente en entrenamiento y evaluación.

**Fuentes de los datos:**

- [Publicación original de HAM10000](https://doi.org/10.1038/sdata.2018.161)
- [Dataset en Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)
- [Copia disponible en Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

La copia de Kaggle descargada informa licencia **CC-BY-NC-SA-4.0**. Se conserva localmente para uso académico no comercial; cualquier redistribución debe respetar la atribución, la condición no comercial y compartir bajo la misma licencia.

## Plan de trabajo

### 1. CNN desde cero

Se construirá una red inicial para clasificar las imágenes en dos grupos:

- **Lesiones malignas o premalignas:** `mel`, `bcc` y `akiec`.
- **Lesiones benignas:** `nv`, `bkl`, `vasc` y `df`.

Esta agrupación es una simplificación experimental. La categoría `akiec` incluye lesiones premalignas y carcinoma intraepitelial, por lo que no se describirá todo el grupo como exclusivamente maligno.

La primera CNN permitirá validar el proceso completo y establecer una línea de base.

### 2. Aprendizaje por transferencia

Se compararán **VGG16, ResNet50 y EfficientNet-B0**, utilizando pesos preentrenados en ImageNet y adaptándolos a la tarea binaria.

### 3. Clasificación multiclase

La arquitectura seleccionada mediante validación se adaptará para distinguir las siete categorías originales del dataset.

## Metodología prevista

Se propone una división aproximada de 70 % para entrenamiento, 15 % para validación y 15 % para prueba, agrupando las imágenes por lesión.

Las imágenes se prepararán a una resolución de 224 × 224 píxeles. Se aplicará aumento de datos únicamente durante el entrenamiento y se evaluarán estrategias para manejar el desbalance, como ponderación de clases y muestreo balanceado.

El conjunto de prueba se reservará para la evaluación final, después de seleccionar las configuraciones con validación.

## Evaluación prevista

Se utilizarán las siguientes métricas y herramientas:

- **Balanced accuracy** como métrica principal.
- **F1 macro** para dar el mismo peso a cada categoría.
- **AUC-ROC** para evaluar la capacidad de discriminación.
- **Sensibilidad por clase** y tasa de falsos negativos.
- **Matrices de confusión** para identificar errores frecuentes.
- **Curvas de aprendizaje** y tiempos de entrenamiento.

La sensibilidad superior al 85 % para melanoma se plantea como una meta exploratoria, no como un resultado obtenido.

## Entregas

| Entrega | Contenido | Estado |
|---|---|---|
| 1 | Propuesta: contexto, objetivos, dataset y metodología | Disponible |
| 2 | Notebooks reproducibles, descarga y experimentos | En desarrollo; datos y particiones listos; entrenamiento completo pendiente |

Las siguientes entregas incorporarán la implementación y los resultados conforme avance el proyecto.

## Próximos pasos

- Completar baseline con el conjunto completo.
- Ejecutar la comparación de transferencia y la clasificación multiclase en una GPU.
- Analizar errores y registrar resultados reales antes de redactar el informe final.

## Alcance

Este proyecto tiene fines académicos. Sus conclusiones dependerán del dataset y del protocolo de evaluación utilizados; no constituye una herramienta validada para diagnóstico clínico.

## Entrega 2 — implementación inicial

Se añadieron cinco notebooks numerados (`00`–`04`) para descarga, exploración y partición agrupada por `lesion_id`, CNN binaria desde cero, comparación binaria con VGG16/ResNet50/EfficientNet-B0 y clasificación multiclase con ResNet50. Las instrucciones de rutas, dependencias y estructura están dentro de cada notebook.

**Estado experimental:** el notebook 01 se ejecutó con HAM10000 y creó particiones por `lesion_id`. Se completaron pruebas diagnósticas cortas en CPU (una época por modelo, muestras pequeñas y equilibradas) para la CNN baseline, VGG16, ResNet50, EfficientNet-B0 y ResNet50 multiclase. Sus métricas están en `results/diagnostic_runs.json`; son verificaciones del flujo y no resultados finales comparables. La corrida completa de baseline se detuvo por el tiempo de cómputo en CPU. Ninguna prueba corta evaluó el conjunto de test; no hay métricas finales de test.

`ENTREGA1.PDF` se conserva sin cambios. El enlace al video se añadirá cuando exista. El dataset descargado está en `data/ham10000/`, ignorado por Git; no redistribuirlo.

## Entorno local

El entorno local de este proyecto utiliza Python 3.12, PyTorch CPU y Jupyter. Para recrearlo en Windows, desde PowerShell en la carpeta del proyecto:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para ejecutar los notebooks desde VS Code/Jupyter, selecciona el kernel `.venv`. El entorno ya creado en esta carpeta usa Python 3.12 y PyTorch CPU; no se detecta CUDA en este equipo. Para abrir Jupyter Lab, activa `.venv` y ejecuta `jupyter lab`. Para entrenar los modelos preentrenados con mayor rapidez, se recomienda Google Colab con GPU.


### Ubicación esperada de HAM10000

En ejecución local, coloca `HAM10000_metadata.csv` y las carpetas de imágenes dentro de `data/ham10000/`. Los notebooks buscan imágenes `.jpg` en `HAM10000_images/`, `HAM10000_images_part_1/` y `HAM10000_images_part_2/`. El notebook 01 guardará las particiones por lesión en `data/splits/`; los notebooks 02–04 leen esos manifiestos. En Colab se mantienen las rutas bajo `/content/`.

### Descarga del dataset

El notebook `00_descarga_datos.ipynb` descarga HAM10000 a `data/ham10000/` mediante Kaggle CLI. Configura primero Kaggle en el equipo con una clave nueva (`%USERPROFILE%\.kaggle\kaggle.json`) o `kaggle auth login`; no guardes tokens en el notebook ni en Git. La dependencia `kaggle` se incluye en `requirements.txt`.

El notebook selecciona automáticamente `%USERPROFILE%\.kaggle` cuando existe; de lo contrario utiliza `data/.kaggle/`. La carpeta `data/` está ignorada por Git para no incluir imágenes ni credenciales.
Para usar el API token actual de Kaggle, guarda uno recién generado en `data/.kaggle/access_token`; el notebook lo detecta automáticamente y `data/` está ignorada por Git. Revoca cualquier token que haya quedado visible en capturas o notebooks compartidos.



## Entrenamiento completo en Colab con GPU

Los notebooks 00–04 fueron revisados para ejecución autónoma en Colab. Consulta [COLAB_GUIDE.md](COLAB_GUIDE.md) para subirlos, configurar Drive y ejecutar el flujo completo. Los helpers visibles proceden de `src/ham10000.py`; `scripts/build_notebooks.py` los mantiene sincronizados.

La versión actual incorpora manifiestos portables por IDs, auditoría de integridad y duplicados, tres estrategias de desbalance para baseline, entrenamiento de cabeza y último bloque en transferencia, selección de arquitectura mediante validación para multiclase, precisión mixta CUDA, progreso por lote/época y reanudación por checkpoint. Las configuraciones y resultados se guardan en Drive bajo `HAM10000_PROJECT/runs/full/colab_v1`. Cambia EXPERIMENT_ID si modificas el protocolo.

La evaluación final del test está desactivada hasta fijar el protocolo en el notebook 04. MODE smoke utiliza otro directorio y bloquea el test. Los diagnósticos de una época guardados anteriormente corresponden a una implementación anterior; no usar sus pesos ni métricas para comparar esta versión.

Se verificaron localmente la sintaxis y el esquema de los cinco notebooks, la auditoría de las 10.015 imágenes, la conservación de las particiones originales y la recuperación de un entrenamiento sintético interrumpido. No se ejecutó esta versión de entrenamiento completo en Colab/GPU y todavía no produce resultados finales del proyecto. Detalle en [reports/COLAB_PREPARACION.md](reports/COLAB_PREPARACION.md).
