# Baseline oficial actual

Actualizado: 2026-06-29

## Run oficial

- Baseline oficial actual: `RUNS/serious_5class_low_60ep_weighted_t14t16`
- Baseline oficial previa: `RUNS/serious_5class_low_60ep_calibrated`
- Config de inferencia fuente: `RUNS/serious_5class_low_60ep_weighted_t14t16/fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json`
- Checkpoint: `best`
- Ruta del checkpoint: `RUNS/serious_5class_low_60ep_weighted_t14t16/checkpoints/best_weights.weights.h5`
- Modo de threshold: `global`
- Threshold global: `0.030`
- NMS: `110 ms`
- Tolerancia onset: `50 ms`
- Split de seleccion: `VAL`
- Split de reporte final: `TEST`
- TEST no se uso para seleccion: `true`

## Esquema de clases

Usar el esquema vigente `vibro_5_toms`:

| Clase | MIDI |
| --- | ---: |
| KD | 35 |
| SD | 38 |
| T12 | 47 |
| T14 | 45 |
| T16 | 43 |

No usar el esquema antiguo `KD, SD, TT, HH, CY`.

## Metricas fulltrack en TEST

| Metrica | Baseline previa | Baseline actual | Delta |
| --- | ---: | ---: | ---: |
| micro-F1 | 0.5716 | 0.5863 | +0.0147 |
| micro-recall | 0.5857 | 0.6196 | +0.0339 |
| macro-F1 | 0.5365 | 0.5855 | +0.0490 |
| T14 F1 | 0.4960 | 0.5425 | +0.0465 |
| T16 F1 | 0.4317 | 0.6349 | +0.2032 |
| SD F1 | 0.5917 | 0.5733 | -0.0184 |

La run weighted queda como baseline oficial porque mejora la metrica principal, TEST fulltrack onset micro-F1, y tambien mejora de forma clara el macro-F1. Las ganancias mas importantes aparecen en las clases debiles T14 y T16, especialmente T16.

## Metricas TEST de la baseline actual

| Metrica | Valor |
| --- | ---: |
| TP | 8657 |
| FP | 6901 |
| FN | 5315 |
| micro-precision | 0.5564 |
| micro-recall | 0.6196 |
| micro-F1 | 0.5863 |
| macro-F1 | 0.5855 |

## Baseline actual por clase

| Clase | MIDI | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KD | 35 | 3640 | 3698 | 1251 | 0.4960 | 0.7442 | 0.5953 |
| SD | 38 | 2533 | 1413 | 2357 | 0.6419 | 0.5180 | 0.5733 |
| T12 | 47 | 820 | 604 | 577 | 0.5758 | 0.5870 | 0.5814 |
| T14 | 45 | 750 | 618 | 647 | 0.5482 | 0.5369 | 0.5425 |
| T16 | 43 | 914 | 568 | 483 | 0.6167 | 0.6543 | 0.6349 |

## Interpretacion

La nueva baseline acepta una pequena baja en precision y SD F1 a cambio de una mejora clara en recall y balance entre clases. Esto coincide con el objetivo de la run con pesos por clase: recuperar mas eventos T14/T16 sin cambiar dataset, generador, DataLoader ni protocolo de entrenamiento.

Para inferencia sobre WAV, usar el checkpoint seleccionado `best`, threshold global `0.030` y NMS `110 ms`, siempre bajo el esquema `KD, SD, T12, T14, T16`.
