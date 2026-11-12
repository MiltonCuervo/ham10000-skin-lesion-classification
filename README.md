# Clasificación de lesiones cutáneas con HAM10000

Proyecto académico de Fundamentos de Deep Learning — Universidad de Antioquia.
**Autor:** Milton Alejandro Cuervo Ramirez.

## Notebooks

Ejecutar en orden. Cada notebook es autónomo e incluye sus funciones.

| Paso | Contenido | Abrir |
|---|---|---|
| 01 | [Exploración de datos y obtención de HAM10000](01_exploracion_de_datos.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/01_exploracion_de_datos.ipynb) |
| 02 | [Preprocesado, particiones y auditoría](02_preprocesado.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/02_preprocesado.ipynb) |
| 03 | [Arquitectura de línea de base](03_arquitectura_linea_base.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/03_arquitectura_linea_base.ipynb) |
| 04 | [Transfer learning binario](04_transfer_learning_binario.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/04_transfer_learning_binario.ipynb) |
| 05 | [Clasificación multiclase y evaluación final](05_clasificacion_multiclase.ipynb) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiltonCuervo/ham10000-skin-lesion-classification/blob/main/05_clasificacion_multiclase.ipynb) |

## Ejecución

1. Abre 01 en Colab. Configura `KAGGLE_API_TOKEN` en Secretos de Colab o utiliza la entrada oculta. Ejecuta descarga y exploración.
2. Ejecuta 02 para preservar o generar las particiones por `lesion_id` y verificar integridad y duplicados.
3. Activa GPU para 03–05 y utiliza el mismo `EXPERIMENT_ID` y `MODE='full'` en todos. Drive conserva checkpoints y resultados; se puede reanudar tras una desconexión.
4. Revisa las selecciones de validación antes de activar `RUN_FINAL_TEST` en 05. No ajustes hiperparámetros a partir del test.

## Estado y entregables

- `ENTREGA1.PDF`: disponible; propuesta original.
- `INFORME_PROYECTO.PDF`: pendiente de los entrenamientos completos, evaluación y análisis. Aún no está incluido.
- **Video en YouTube: pendiente de grabación y publicación. El enlace debe añadirse aquí antes de entregar.**

Datos y particiones verificados: 10.015 imágenes y división 7.002 / 1.519 / 1.494 sin lesiones compartidas. Se realizaron comprobaciones breves del código en CPU. El entrenamiento completo de esta versión en GPU y las métricas finales siguen pendientes. Las corridas breves no se presentan como resultados finales.

## Datos y alcance

HAM10000: Tschandl, Rosendahl y Kittler (2018), [publicación original](https://doi.org/10.1038/sdata.2018.161). [Copia en Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), que declara CC-BY-NC-SA-4.0. Los datos se descargan desde 01 y no se incluyen en el repositorio.

Clasificación binaria: `mel`, `bcc` y `akiec` constituyen el grupo maligno o premaligno; multiclase utiliza las siete categorías originales. El proyecto tiene fines académicos y no constituye una herramienta validada para diagnóstico clínico.
