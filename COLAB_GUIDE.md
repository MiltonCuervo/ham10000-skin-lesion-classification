# HAM10000: ejecución completa en Colab con GPU

Estado: notebooks preparados y verificados localmente; los entrenamientos completos en GPU quedan por ejecutar. Las pruebas cortas anteriores son diagnósticas y corresponden a una versión anterior del código. No reutilices sus pesos como resultados de esta versión.

## Preparación desde GitHub

1. Abre el README del repositorio y pulsa **Abrir en Colab** en el notebook 00. Continúa con los botones 01–04 en ese orden.
2. Cada notebook incorpora sus helpers. No necesitas descargar el ZIP, clonar el repositorio ni subir el código a Drive.
3. Autoriza el montaje de Drive: se crea `Mi unidad/HAM10000_PROJECT` para caché de datos, particiones y resultados. Las imágenes se extraen al disco local del runtime.
4. Usa el mismo `EXPERIMENT_ID='colab_v1'`, `MODE='full'` y la misma carpeta `PERSIST` en todos los notebooks. Si no existen manifiestos en Drive, 01 genera las particiones con la semilla fija; si existen, las valida y reutiliza. MODE smoke utiliza otro directorio y bloquea el test.
5. Configura `KAGGLE_API_TOKEN` en Secretos de Colab con acceso al notebook para descargar desde Kaggle. También existe entrada oculta. No escribas la clave en celdas ni salidas.
6. En 02–04 selecciona Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU.

El ZIP continúa disponible como alternativa local. Si lo usas, extrae su contenido en Drive bajo HAM10000_PROJECT, conservando las carpetas como directorios directos.

## Orden de ejecución

- **00 — Datos:** descarga Kaggle y extrae los JPG a `/content/ham10000`. Con `CACHE_DATASET_ZIP=True` conserva una copia ZIP en Drive para reconexiones. Necesitas espacio para un ZIP de unos 5,2 GB, los archivos extraídos y los checkpoints. Si ya tienes el ZIP oficial, puedes subirlo a `HAM10000_PROJECT/dataset_cache/skin-cancer-mnist-ham10000.zip` y evitar otra descarga.
- **01 — Exploración:** valida metadatos, muestra desbalance y ejemplos, preserva las particiones, verifica todos los JPG y calcula SHA-256. Falla si hay corrupción, ausencia o duplicados exactos entre particiones. Revisa `audit/duplicates.csv` aunque las copias estén dentro de una misma partición. Una incidencia requiere revisión metodológica; no se eliminan ni trasladan imágenes automáticamente.
- **02 — Baseline:** activa GPU en Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU. Compara pérdida sin ponderación, pérdida ponderada y muestreo balanceado, hasta 20 épocas por variante. Early stopping: 5 épocas sin mejora de balanced accuracy de validación.
- **03 — Transfer:** VGG16, ResNet50 y EfficientNet-B0; 5 épocas de cabeza y hasta 15 de último bloque. Usa batch 16 común y pérdida ponderada. VGG16 sustituye la cabeza grande original por average pooling global, dropout y capa lineal. Selecciona arquitectura por balanced accuracy de validación, con F1 macro como desempate.
- **04 — Multiclase:** toma la arquitectura elegida en 03, reinicia desde ImageNet, cambia la cabeza a siete clases y entrena 5 épocas de cabeza más hasta 20 del último bloque. El bloque final de evaluación del test permanece desactivado hasta que fijes el protocolo.

El presupuesto de épocas es un punto de partida explícito para los experimentos, no una promesa de convergencia. Registra cualquier cambio con un EXPERIMENT_ID nuevo. No selecciones configuraciones por los resultados del test.

## Desconexiones y reanudación

Drive conserva `splits/`, `audit/` y `runs/{full|smoke}/{EXPERIMENT_ID}/`. Después de reconectar, ejecuta las celdas desde el comienzo. La celda de restauración extrae el ZIP al disco local si hace falta. `last.pt` recupera modelo, optimizador, AMP, historial y época; `best.pt` conserva el modelo elegido por validación. Si la desconexión ocurre dentro de una época, se repite esa época completa. Los CSV nunca contienen rutas absolutas de Windows.

Una corrida completa se detecta y reutiliza. Cada ejecución registra configuración, versiones, dispositivo, semilla y huellas de particiones y auditoría. Una configuración distinta sobre el mismo directorio se rechaza. Cambia EXPERIMENT_ID si ajustas batch, LR, etapas, librerías o estrategia. Las semillas y los algoritmos deterministas ayudan a repetir las corridas; diferentes versiones, GPUs o kernels CUDA pueden producir pequeñas diferencias.

## Memoria de GPU

Si aparece CUDA out of memory, reinicia el runtime para liberar memoria, reduce BATCH_SIZE y usa otro EXPERIMENT_ID. Mantén un batch común entre las tres arquitecturas de transferencia. El código utiliza precisión mixta con CUDA y libera cada modelo al terminar. No instala ni reemplaza las versiones CUDA de torch/torchvision que trae Colab; sí registra las versiones usadas.

## Resultados y test final

Cada corrida guarda `config.json`, `last.pt`, `best.pt`, `history.csv`, `learning_curves.png` y `validation/` con métricas, predicciones por imagen, reporte por clase, matriz de confusión y ROC. Se calculan balanced accuracy, F1 macro, AUC (binaria u OvR macro), sensibilidad y tasa de falsos negativos por clase. AUC se registra como null si no puede calcularse.

Tras revisar validación y completar el presupuesto o early stopping, fija las selecciones y cambia RUN_FINAL_TEST a True en la última celda de 04. Se bloquea el protocolo con las firmas seleccionadas y se evalúan la mejor baseline, el mejor transfer binario y el multiclase. Los resultados existentes se reutilizan; cambiar la selección después de fijar el protocolo queda bloqueado. No uses estas métricas para reabrir la búsqueda de hiperparámetros.

No confundir los checkpoints diagnósticos anteriores con `runs/full/colab_v1`. No se han obtenido todavía métricas finales de esta implementación. Conserva ENTREGA1.PDF y redacta INFORME_PROYECTO.PDF y el enlace al video después de ejecutar y analizar los experimentos completos.

## Fuentes y mantenimiento

Datos: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 ; publicación original: https://doi.org/10.1038/sdata.2018.161 . La copia de Kaggle declara CC-BY-NC-SA-4.0; conserva atribución y condiciones de uso.

El código común está en `src/ham10000.py`. Después de editarlo, ejecuta `python scripts/build_notebooks.py` para regenerar los helpers visibles de los cinco notebooks. El generador borra salidas; respalda notebooks ejecutados antes de regenerar. El paquete excluye datos, credenciales, entornos y resultados históricos.
