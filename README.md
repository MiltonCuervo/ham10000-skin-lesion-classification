# Clasificación de lesiones cutáneas con HAM10000

Proyecto académico de Fundamentos de Deep Learning — Universidad de Antioquia.
**Autor:** Milton Alejandro Cuervo Ramirez.

## Ejecución local

Desde la carpeta del proyecto abre Jupyter:

```powershell
& "$env:USERPROFILE\Documents\Codex\environments\ham10000\.venv\Scripts\python.exe" -m jupyterlab
```

El entorno Python está separado de la carpeta de entrega. Selecciona el kernel **Python (HAM10000 local GPU)** y ejecuta los notebooks en orden. El entorno local está instalado con Python 3.12, PyTorch 2.6.0 + CUDA 11.8 y torchvision 0.21.0. Se comprobó la GTX 1050 de 3 GB con operaciones reales. Batch predeterminado: 8; precisión FP32 en esta GPU. Los notebooks también aceptan CPU, aunque será más lenta.

| Paso | Contenido | Alternativa en Colab |
|---|---|---|
| 01 | [Exploración y datos](01_exploracion_de_datos.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/01_exploracion_de_datos.ipynb) |
| 02 | [Preprocesado y particiones](02_preprocesado.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/02_preprocesado.ipynb) |
| 03 | [CNN baseline](03_arquitectura_linea_base.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/03_arquitectura_linea_base.ipynb) |
| 04 | [Transfer learning binario](04_transfer_learning_binario.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/04_transfer_learning_binario.ipynb) |
| 05 | [Multiclase y ajuste fino](05_clasificacion_multiclase.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/05_clasificacion_multiclase.ipynb) |

## Resultados finales

Los entrenamientos completos y la evaluación final terminaron. Cada modelo seleccionado se evaluó sobre las mismas **1.494 imágenes de test**, sin lesiones compartidas con entrenamiento ni validación. La elección de modelo y checkpoint se realizó previamente mediante validación.

| Tarea y modelo seleccionado | Balanced accuracy | F1 macro | AUC-ROC |
|---|---:|---:|---:|
| Binaria: CNN con pesos | 0.7865 | 0.6687 | 0.8649 |
| Binaria: ResNet50 | 0.7585 | 0.6973 | 0.8483 |
| Siete clases: ResNet50 con ajuste fino | 0.7123 | 0.5814 | 0.9429 |

La AUC multiclase corresponde a OvR macro y no se compara directamente con la AUC binaria. Las tablas, curvas de aprendizaje, matrices de confusión y ROC se pueden consultar en los notebooks **03, 04 y 05**, que conservan sus salidas ejecutadas.

En la tarea binaria, la CNN obtuvo mayor balanced accuracy y sensibilidad del grupo maligno o premaligno (94,03 %), con 19 falsos negativos y 432 falsos positivos. ResNet50 obtuvo mayor F1 macro, con 73 falsos negativos y 298 falsos positivos. Se conservó la selección realizada por validación; no se cambió la arquitectura después de consultar test.

En multiclase, la sensibilidad para melanoma fue **61,50 % (115/187)**: hubo 72 imágenes de melanoma asignadas a otra categoría. La precisión de dermatofibroma fue 12,50 %, con solo 7 ejemplos reales de esa clase. Estos errores y el tamaño reducido de algunas clases limitan las conclusiones, aunque la AUC macro sea alta.

Las métricas fueron recalculadas desde los CSV de predicciones y coincidieron con los reportes. Se comprobaron las firmas de los checkpoints, los manifiestos y las selecciones fijadas antes de test. Son resultados de una partición reservada y una semilla; no representan validación clínica ni estiman variabilidad entre particiones.

## Flujo de trabajo

- **01:** utiliza los datos locales; descarga desde Kaggle únicamente si faltan. Muestra desbalance, metadatos y ejemplos.
- **02:** conserva las particiones 70/15/15 por `lesion_id`, verifica originales y crea una caché RGB de 224×224 en PNG. La caché evita repetir decodificación y redimensionado en cada época. Los originales no se modifican.
- **03:** CNN pequeña desde cero, 10 épocas, comparación sin pesos y con pesos de clase.
- **04:** VGG16, ResNet50 y EfficientNet-B0, un modelo a la vez; backbone congelado, 5 épocas de clasificador y pérdida ponderada. VGG16 usa pooling global y una cabeza pequeña para la memoria local.
- **05:** arquitectura seleccionada mediante validación; clasificador de siete clases (8 épocas) y último bloque más clasificador (5 épocas, LR menor).

Cada notebook incluye solo las funciones que necesita. Los checkpoints, historiales, curvas y reportes se guardan en `results/full/local_v2/`. Se reanuda desde la última época completa. Cambia `RUN_ID` al modificar la configuración. El aumento se aplica solo a train; los pesos usan frecuencias de train.

`MODE='smoke'` permite una comprobación mínima y guarda sus artefactos por separado. En las ejecuciones completas publicadas se fijaron las configuraciones y se activó `RUN_FINAL_TEST` para la evaluación final. Durante nuevos experimentos mantén el test reservado y utiliza un RUN_ID distinto. La selección de arquitectura y checkpoints usa validación.

## Estado y entregables

Se ejecutaron los cinco notebooks. Se mantuvo la división 7.002 / 1.519 / 1.494 de las 10.015 imágenes, sin lesiones compartidas. Se completaron las dos variantes de baseline (10 épocas cada una), los tres clasificadores de transferencia (5 épocas cada uno) y la multiclase (8 épocas de cabeza y 5 de ajuste fino). Las tres evaluaciones finales y sus salidas están disponibles en los notebooks.

- `ENTREGA1.PDF`: disponible, propuesta original.
- [INFORME_PROYECTO.PDF](INFORME_PROYECTO.PDF): informe final de ocho páginas con metodología, resultados reales verificados, curvas de aprendizaje, matrices de confusión, análisis de errores, limitaciones y referencias.
- **Video en YouTube: pendiente de grabación/publicación. Añadir el enlace aquí antes de entregar.**

Los datos originales, credenciales, entornos y checkpoints no se incluyen en GitHub. Las métricas y figuras finales sí se publican dentro de los notebooks y en este resumen. El repositorio conserva solamente los notebooks numerados, informes disponibles, README y reglas de exclusión.

## Dataset y alcance

HAM10000: Tschandl, Rosendahl y Kittler (2018), [publicación original](https://doi.org/10.1038/sdata.2018.161). La [copia de Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) declara CC-BY-NC-SA-4.0; respetar atribución y uso no comercial.

El grupo positivo binario contiene `mel`, `bcc` y `akiec` (malignas o premalignas). Multiclase utiliza las siete categorías originales. El proyecto tiene fines académicos y no es una herramienta validada de diagnóstico clínico.
