# Run analysis summary: serious_5class_low_60ep_focal_soft_toms_v3

Fecha de resumen: 2026-07-16

Este documento resume la run `RUNS/serious_5class_low_60ep_focal_soft_toms_v3` usando artefactos ya existentes de entrenamiento, calibracion full-track + NMS, comparacion disponible y analisis de errores. No se recalculo entrenamiento, no se modificaron datasets, generador, DataLoader, checkpoints ni metricas.

## 1. Identificacion de la run

| Campo | Valor |
| --- | --- |
| Run | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3` |
| Dataset | `DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` |
| Esquema | `vibro_5_toms`: KD=35, SD=38, T12=47, T14=45, T16=43 |
| Arquitectura | `Frame_RNN` |
| Epochs | 60 |
| Steps por epoch | 600 |
| Validation steps | 150 |
| Batch size | 1 |
| Training sequence | 64 |
| Context | 9 |
| Sample rate | 100 |
| Loss type | `focal` |
| Focal gamma | 1.0 |
| Focal alpha | 0.25 |
| Class weights | KD=1.0, SD=1.0, T12=1.2, T14=1.4, T16=1.6 |

Artefactos de entrenamiento usados:

- `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/config.json`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/history.csv`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/history.json`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/dataset_event_counts.csv`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/dataset_event_counts.json`

Checkpoints disponibles:

| Checkpoint | Archivo | Tamano aprox. | Seleccionado |
| --- | --- | ---: | --- |
| `best` | `checkpoints/best_weights.weights.h5` | 1.62 MB | No |
| `final` | `checkpoints/final_weights.weights.h5` | 1.62 MB | Si |
| metadata | `checkpoints/checkpoint_metadata.json` | 426 B | Si |

## 2. Calibracion full-track + NMS

La calibracion oficial de la run esta en:

`RUNS/serious_5class_low_60ep_focal_soft_toms_v3/fulltrack_calibration`

Archivo principal para inferencia:

`RUNS/serious_5class_low_60ep_focal_soft_toms_v3/fulltrack_calibration/fulltrack_selected_inference_config.json`

| Campo | Valor |
| --- | --- |
| Checkpoint seleccionado por VAL | `final` |
| Threshold seleccionado | 0.04 |
| Modo de threshold | `global` |
| Thresholds por clase | No hay thresholds por clase; el analisis global usa 0.04 para todas las clases |
| NMS seleccionado | 150 ms |
| Tolerancia onset | 50 ms |
| Split de seleccion | `VAL` |
| Split de reporte final | `TEST` |
| TEST usado para seleccion | No (`test_not_used_for_selection=true`) |
| Dataset resuelto | `DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` |

Metricas TEST con la configuracion congelada desde VAL:

| TP | FP | FN | Precision | Recall | Micro-F1 | Macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7788 | 6726 | 6184 | 0.5366 | 0.5574 | 0.5468 | 0.5408 |

## 3. Comparacion contra baseline

Baseline oficial indicada para esta revision:

`RUNS/serious_5class_low_60ep_focal_soft_toms_v2`

No existe la carpeta de comparacion solicitada:

`RUNS/COMPARISONS/serious_5class_low_60ep_focal_soft_toms_v3_vs_serious_5class_low_60ep_focal_soft_toms_v2`

Por trazabilidad, la tabla siguiente compara directamente los artefactos `fulltrack_selected_inference_config.json` de `focal_soft_toms_v2` y `focal_soft_toms_v3`. No es una comparacion generada por `compare_fulltrack_runs.py`; es una lectura resumida de configs ya existentes.

| Metrica TEST | Baseline v2 | Candidata v3 | Delta v3-v2 |
| --- | ---: | ---: | ---: |
| Micro-precision | 0.5943 | 0.5366 | -0.0577 |
| Micro-recall | 0.6431 | 0.5574 | -0.0857 |
| Micro-F1 | 0.6177 | 0.5468 | -0.0709 |
| Macro-F1 | 0.5899 | 0.5408 | -0.0491 |
| TP | 8985 | 7788 | -1197 |
| FP | 6134 | 6726 | +592 |
| FN | 4987 | 6184 | +1197 |

Configuraciones comparadas:

| Run | Checkpoint | Threshold | NMS | Tolerancia |
| --- | --- | ---: | ---: | ---: |
| `focal_soft_toms_v2` | `best` | 0.05 | 110 ms | 50 ms |
| `focal_soft_toms_v3` | `final` | 0.04 | 150 ms | 50 ms |

F1 por clase:

| Clase | MIDI | F1 baseline v2 | F1 candidata v3 | Delta | TP base/cand | FP base/cand | FN base/cand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KD | 35 | 0.6417 | 0.5708 | -0.0709 | 3819/3210 | 3192/3146 | 1072/1681 |
| SD | 38 | 0.6324 | 0.5291 | -0.1033 | 3166/2399 | 1957/1779 | 1724/2491 |
| T12 | 47 | 0.5353 | 0.5192 | -0.0161 | 640/797 | 354/876 | 757/600 |
| T14 | 45 | 0.5347 | 0.5012 | -0.0335 | 667/648 | 431/541 | 730/749 |
| T16 | 43 | 0.6052 | 0.5837 | -0.0215 | 693/734 | 200/384 | 704/663 |

Lectura breve: la candidata v3 empeora todas las clases en F1 frente a v2. La degradacion mas fuerte aparece en SD (-0.1033) y KD (-0.0709). En toms la caida es menor pero consistente: T12 (-0.0161), T14 (-0.0335), T16 (-0.0215). T12 y T16 aumentan TP y reducen FN, pero pagan con incrementos fuertes de FP, especialmente T12 (+522 FP) y T16 (+184 FP).

## 4. Analisis de errores

Analisis usado:

`RUNS/serious_5class_low_60ep_focal_soft_toms_v3/error_analysis_fulltrack_global`

Configuracion del analisis:

| Campo | Valor |
| --- | --- |
| Split | `TEST` |
| Threshold source | `fulltrack_inference_config` |
| Threshold mode | `global` |
| Threshold | 0.04 |
| NMS | 150 ms |
| Tolerancia onset | 50 ms |
| Tracks analizados | 1200 / 1200 |
| Dataset root source | `cli_override` |

Clases con mas falsos positivos:

| Clase | FP | TP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KD | 3146 | 3210 | 1681 | 0.5050 | 0.6563 | 0.5708 |
| SD | 1779 | 2399 | 2491 | 0.5742 | 0.4906 | 0.5291 |
| T12 | 876 | 797 | 600 | 0.4764 | 0.5705 | 0.5192 |
| T14 | 541 | 648 | 749 | 0.5450 | 0.4639 | 0.5012 |
| T16 | 384 | 734 | 663 | 0.6565 | 0.5254 | 0.5837 |

Clases con mas falsos negativos:

| Clase | FN | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SD | 2491 | 2399 | 1779 | 0.5742 | 0.4906 | 0.5291 |
| KD | 1681 | 3210 | 3146 | 0.5050 | 0.6563 | 0.5708 |
| T14 | 749 | 648 | 541 | 0.5450 | 0.4639 | 0.5012 |
| T16 | 663 | 734 | 384 | 0.6565 | 0.5254 | 0.5837 |
| T12 | 600 | 797 | 876 | 0.4764 | 0.5705 | 0.5192 |

Peor clase por F1:

- `T14`, con F1=0.5012, precision=0.5450, recall=0.4639, TP=648, FP=541, FN=749.

Patrones por perfil con menor F1:

| Perfil | Clase | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `basic_groove` | T14 | 147 | 129 | 217 | 0.5326 | 0.4038 | 0.4594 |
| `fill_heavy` | T12 | 122 | 158 | 108 | 0.4357 | 0.5304 | 0.4784 |
| `fast_dense` | T14 | 68 | 63 | 74 | 0.5191 | 0.4789 | 0.4982 |
| `standard_rock` | T14 | 199 | 167 | 225 | 0.5437 | 0.4693 | 0.5038 |
| `fast_dense` | T12 | 83 | 95 | 67 | 0.4663 | 0.5533 | 0.5061 |

Peores tracks por F1:

| Track | Perfil | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_standard_rock_000680_129bpm` | standard_rock | 1 | 9 | 9 | 0.1000 | 0.1000 | 0.1000 |
| `test_standard_rock_000126_122bpm` | standard_rock | 1 | 6 | 10 | 0.1429 | 0.0909 | 0.1111 |
| `test_basic_groove_001134_104bpm` | basic_groove | 2 | 8 | 10 | 0.2000 | 0.1667 | 0.1818 |
| `test_standard_rock_001090_109bpm` | standard_rock | 2 | 6 | 10 | 0.2500 | 0.1667 | 0.2000 |
| `test_basic_groove_001014_104bpm` | basic_groove | 2 | 8 | 8 | 0.2000 | 0.2000 | 0.2000 |

Consistencia de ground truth:

- `ground_truth_count_check.json`: `status=match`.
- Conteos desde anotaciones, `dataset_event_counts.csv` y analizador coinciden exactamente:
  - KD=4891
  - SD=4890
  - T12=1397
  - T14=1397
  - T16=1397
- `track_coverage_check.json`: `status=match`.
- Tracks anotados: 1200.
- Tracks usados por el analizador: 1200.
- Duplicados: ninguno.
- Tracks faltantes: ninguno.
- Tracks extra: ninguno.
- Tracks con mismatch de ground truth: ninguno.

## 5. Interpretacion

La run v3 no mejora la baseline oficial `focal_soft_toms_v2`. La degradacion es global: baja precision, baja recall, baja micro-F1 y baja macro-F1. El cambio de configuracion seleccionada desde VAL (`final`, threshold 0.04, NMS 150 ms) no se traduce en una mejor configuracion TEST que la baseline v2 (`best`, threshold 0.05, NMS 110 ms).

Que empeora:

- Micro-F1 cae de 0.6177 a 0.5468 (-0.0709).
- Macro-F1 cae de 0.5899 a 0.5408 (-0.0491).
- TP baja en 1197 eventos.
- FP sube en 592 eventos.
- FN sube en 1197 eventos.
- SD es la clase mas afectada por F1 (-0.1033), con una perdida importante de recall.
- KD tambien cae de forma fuerte (-0.0709).
- T14 queda como peor clase absoluta por F1 (0.5012).

Que mejora parcialmente:

- T12 aumenta TP y recall respecto a v2, pero el costo en FP es alto; su precision baja de 0.6439 a 0.4764.
- T16 tambien aumenta TP y recall, pero incrementa FP y termina con F1 inferior a v2.

Sobre toms:

- No hay una degradacion catastrofica aislada en toms, pero si una caida consistente en F1 para T12, T14 y T16.
- T14 sigue siendo el punto mas fragil: baja recall y queda como peor F1 global.
- T12 y T16 muestran un trade-off de mayor sensibilidad con exceso de falsos positivos.

Aporte experimental:

Esta run es util como ablation de focal loss suave con class weights KD=1.0, SD=1.0, T12=1.2, T14=1.4, T16=1.6. El resultado sugiere que esta variante desplaza el comportamiento hacia una deteccion menos estable: no obtiene el beneficio global de v2 y no corrige de forma neta el problema de toms. Es defendible como experimento negativo: ayuda a delimitar que esta combinacion de focal loss, pesos y seleccion `final + threshold 0.04 + NMS 150 ms` no debe reemplazar la baseline.

## 6. Decision recomendada

**Descartar como baseline.**

La run `serious_5class_low_60ep_focal_soft_toms_v3` no reemplaza a `serious_5class_low_60ep_focal_soft_toms_v2`: pierde en metricas agregadas, pierde en todas las clases por F1, y no ofrece una mejora especifica suficientemente clara en toms que compense la caida global.

## 7. Siguiente accion sugerida

Recomendacion principal:

- Mantener `RUNS/serious_5class_low_60ep_focal_soft_toms_v2` como baseline oficial.

Acciones utiles:

- Probar la baseline actual `focal_soft_toms_v2` en `drumTranscriptor_vibro.ipynb` con WAV real antes de dedicar mas tiempo a v3.
- No usar v3 como modelo por defecto de inferencia.
- Si se entrena una nueva variante, evitar repetir exactamente la combinacion de v3; explorar una de estas direcciones:
  - ajustar class weights de toms sin reducir tanto la estabilidad de KD/SD;
  - probar thresholds por clase en vez de threshold global;
  - revisar una grilla NMS menos agresiva que 150 ms para evitar perder eventos cercanos;
  - conservar la seleccion por VAL, pero comparar explicitamente `best` vs `final` cuando el VAL seleccione `final` y el TEST caiga mucho.

## 8. Artefactos faltantes

Faltante esperado por la solicitud:

- `RUNS/COMPARISONS/serious_5class_low_60ep_focal_soft_toms_v3_vs_serious_5class_low_60ep_focal_soft_toms_v2`

Artefacto alternativo existente, pero contra otra baseline:

- `RUNS/COMPARISONS/focal_soft_toms_v3_vs_weighted_t14t16`

Por este motivo, la comparacion contra `focal_soft_toms_v2` en este documento se hizo leyendo directamente:

- `RUNS/serious_5class_low_60ep_focal_soft_toms_v2/fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/fulltrack_calibration/fulltrack_selected_inference_config.json`

No se genero una nueva comparacion y no se recalcularon metricas.

## 9. Paths de trazabilidad

| Tipo | Path |
| --- | --- |
| Run | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3` |
| Config entrenamiento | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/config.json` |
| Metadata checkpoint | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/checkpoints/checkpoint_metadata.json` |
| Checkpoint seleccionado | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/checkpoints/final_weights.weights.h5` |
| Calibracion | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/fulltrack_calibration` |
| Inference config | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/fulltrack_calibration/fulltrack_selected_inference_config.json` |
| Analisis de errores | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/error_analysis_fulltrack_global` |
| Error summary fuente | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/error_analysis_fulltrack_global/error_summary.md` |
| Ground truth check | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/error_analysis_fulltrack_global/ground_truth_count_check.json` |
| Track coverage check | `RUNS/serious_5class_low_60ep_focal_soft_toms_v3/error_analysis_fulltrack_global/track_coverage_check.json` |
| Baseline usada para lectura comparativa | `RUNS/serious_5class_low_60ep_focal_soft_toms_v2/fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json` |
