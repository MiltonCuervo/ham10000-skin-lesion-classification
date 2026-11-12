# Comprobación de preparación para Colab — 9 de noviembre de 2026

Se revisaron los notebooks 00–04 y se regeneraron sin salidas previas. La copia anterior de los notebooks y README se respalda antes de la integración.

Comprobaciones locales completadas:

- Esquema Jupyter y sintaxis Python de todas las celdas.
- Helpers incrustados sincronizados con src/ham10000.py.
- Dataset real: 10.015 imágenes, metadatos únicos y 7.470 lesiones.
- Se preservaron las particiones anteriores: train 7.002, valid 1.519 y test 1.494; sin lesion_id ni image_id compartidos.
- Integridad PIL y SHA-256 de todos los JPG: sin imágenes ausentes/corruptas ni duplicados exactos entre particiones.
- Entrenamiento sintético breve de baseline, checkpoint, reanudación de corrida terminada y rechazo de configuración incompatible.
- Desconexión simulada entre épocas y reanudación: parámetros finales idénticos en CPU a la corrida continua.
- Activación progresiva de parámetros de cabeza y último bloque en VGG16, ResNet50 y EfficientNet-B0.
- Cálculo de métricas, muestreo ponderado y bloqueo de test para modo smoke.

Las comprobaciones sintéticas validan el funcionamiento del código; no son resultados científicos. El test de HAM10000 no se evaluó. El camino CUDA/AMP y el entorno remoto de Colab no se ejecutaron desde esta sesión; deberán comprobarse al iniciar la corrida real. Los entrenamientos completos y sus resultados, el informe PDF y el video continúan pendientes.

La versión registra parámetros, librerías, dispositivo y huellas de datos. La reproducibilidad entre hardware/versiones distintos puede presentar pequeñas diferencias aun con semillas y algoritmos deterministas.

Comprobación adicional de principio a fin: los notebooks 02, 03 y 04 se ejecutaron en CPU en modo smoke con muestras mínimas de los datos reales. Pasaron las tres estrategias de baseline, las dos etapas de las tres arquitecturas de transferencia y la multiclase con arquitectura elegida mediante validación. Se comprobó que no se generaron FINAL_PROTOCOL ni métricas de test. Las salidas se guardaron bajo runs/smoke, separadas de runs/full.
