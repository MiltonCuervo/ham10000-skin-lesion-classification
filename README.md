# Clasificación de lesiones cutáneas con HAM10000

Proyecto académico de **Fundamentos de Deep Learning** de la Universidad de Antioquia. Se propone comparar redes neuronales convolucionales para clasificar imágenes dermatoscópicas del dataset HAM10000 utilizando PyTorch.

**Autor:** Milton Alejandro Cuervo Ramirez  
**Programa:** Ingeniería de Sistemas  
**Profesores:** Raúl Ramos y Julián Arias  
**Fecha:** Septiembre de 2026

## Estado actual

El proyecto se encuentra en la etapa de formulación. **La única entrega disponible es la propuesta inicial**, donde se describen el problema, el dataset, los modelos que se evaluarán y la metodología prevista.

Todavía no se han implementado ni entrenado modelos, y no se cuenta con resultados experimentales.

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

Los datos tienen licencia **CC BY-NC 4.0**. Su uso requiere atribución y está restringido a fines no comerciales.

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

Las siguientes entregas incorporarán la implementación y los resultados conforme avance el proyecto.

## Próximos pasos

- Obtener y revisar las imágenes y los metadatos.
- Realizar el análisis exploratorio.
- Definir las particiones por lesión.
- Implementar y evaluar la CNN inicial.

## Alcance

Este proyecto tiene fines académicos. Sus conclusiones dependerán del dataset y del protocolo de evaluación utilizados; no constituye una herramienta validada para diagnóstico clínico.
