# Auditoria enfocada de outputs en RUNS

Fecha: 2026-07-14

Alcance: auditoria documental de outputs en `RUNS/`, enfocada en baseline/candidatas actuales para inferencia sobre WAV reales. No se borraron, movieron, renombraron ni modificaron archivos de `RUNS/`. No se tocaron datasets, checkpoints, resultados, scripts `.py`, commits ni push.

Contexto vigente:

- Proyecto ADTOF adaptado a senales vibratorias/piezos.
- Esquema `vibro_5_toms`: KD=35, SD=38, T12=47, T14=45, T16=43.
- Metrica principal: onset-based F1 con tolerancia temporal.
- Dataset principal actual: `DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`.
- Scripts reorganizados actuales:
  - `SCRIPTS/training/train_custom_split.py`
  - `SCRIPTS/calibration/calibrate_fulltrack_thresholds.py`
  - `SCRIPTS/analysis/analyze_run_errors.py`
  - `SCRIPTS/analysis/compare_fulltrack_runs.py`

## 1. Runs auditadas

Se priorizaron runs serias recientes y comparaciones asociadas. Las smoke runs se clasifican solo como descartables/revisables, sin auditoria exhaustiva.

| Run | Rol observado | Estado para inferencia WAV real |
| --- | --- | --- |
| `serious_5class_low_60ep_weighted_t14t16` | Baseline oficial actual segun `RUNS/current_official_baseline.json`. | Usable ahora. Mantener paquete minimo de inferencia. |
| `serious_5class_low_60ep_focal_soft_toms_v2` | Candidata reciente que mejora micro-F1 global frente a baseline oficial. | Usable como candidata; REVISAR por caida en T12/T14/T16. |
| `serious_5class_low_60ep_calibrated` | Baseline previa. | Mantener para trazabilidad y comparacion historica. |
| `serious_5class_low_60ep_weighted_sd_toms_v1` | Candidata calibrada, rendimiento casi igual a baseline oficial. | REVISAR; usable, pero no mejora la baseline. |
| `serious_5class_low_60ep_weighted_sd_light_keep_toms_v2` | Candidata comparada, pero sin `fulltrack_selected_inference_config.json` en su carpeta. | REVISAR antes de inferir. |
| `serious_5class_low_60ep_weighted_balanced_v2` | Carpeta con calibracion full-track, sin checkpoints propios observados. | REVISAR: parece calibracion/output parcial. |
| `serious_5class_low_60ep_weighted_balanced_v3` | Carpeta con checkpoints, sin inference config full-track observada. | REVISAR antes de inferir. |
| `serious_5class_low_60ep_focal_toms_v1` | Candidata calibrada, peor que baseline oficial en micro-F1. | Secundaria; mantener trazabilidad si se conserva la comparacion. |
| `baseline_5class_low_30ep_calibrated` | Baseline anterior de 30 epocas. | Historica; no prioritaria para WAV reales actuales. |

## 2. Configs de inferencia encontradas

| Run | Config seleccionada | Checkpoint | Threshold | NMS ms | TEST micro-F1 | TEST macro-F1 | Lectura |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `serious_5class_low_60ep_calibrated` | `fulltrack_calibration_f1_nms_refined/fulltrack_selected_inference_config.json` | `best` | 0.02 | 125 | 0.5716 | 0.5365 | Baseline previa. |
| `serious_5class_low_60ep_weighted_t14t16` | `fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json` | `best` | 0.03 | 110 | 0.5863 | 0.5855 | Baseline oficial actual. |
| `serious_5class_low_60ep_weighted_sd_toms_v1` | `fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json` | `final` | 0.02 | 110 | 0.5855 | 0.5622 | Muy cercana, no supera baseline. |
| `serious_5class_low_60ep_weighted_balanced_v2` | `fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json` | `best` | 0.03 | 90 | 0.5813 | 0.5734 | REVISAR: no se observaron checkpoints en esa carpeta. |
| `serious_5class_low_60ep_focal_toms_v1` | `fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json` | `final` | 0.075 | 150 | 0.5745 | 0.5567 | Secundaria. |
| `serious_5class_low_60ep_focal_soft_toms_v2` | `fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json` | `best` | 0.05 | 110 | 0.6177 | 0.5899 | Candidata fuerte global; revisar toms. |

No se encontro `fulltrack_selected_inference_config.json` en:

- `baseline_5class_low_30ep_calibrated`
- `serious_5class_low_60ep_weighted_sd_light_keep_toms_v2`
- `serious_5class_low_60ep_weighted_balanced_v3`

## 3. Comparaciones relevantes

| Comparacion | Resultado | Delta micro-F1 | Lectura |
| --- | --- | ---: | --- |
| `weighted_t14t16_vs_baseline` | `weighted_t14t16` mejora a `serious_5class_low_60ep_calibrated`. | +0.0147 | Justifica baseline oficial actual. |
| `focal_soft_toms_v2_vs_weighted_t14t16` | `focal_soft_toms_v2` mejora micro-F1 global. | +0.0314 | Candidata fuerte para metrica global; empeora T12/T14/T16 frente a baseline. |
| `focal_toms_v1_vs_weighted_t14t16` | Peor que baseline oficial. | -0.0118 | Secundaria. |
| `weighted_sd_toms_v1_vs_weighted_t14t16` | Ligeramente peor que baseline oficial. | -0.0008 | Secundaria/REVISAR si interesa balance SD-toms. |
| `weighted_sd_toms_v2_vs_weighted_t14t16` | Ligeramente peor que baseline oficial. | -0.0008 | REVISAR: candidata sin inference config visible en su carpeta. |
| `weighted_balanced_v2_vs_weighted_t14t16` | Peor que baseline oficial. | -0.0051 | REVISAR: carpeta de comparacion dice `v2`, JSON apunta a candidata `weighted_balanced_v3`, mientras la inference config observada esta en `weighted_balanced_v2`. |

Detalle importante de `focal_soft_toms_v2`: mejora micro-precision, micro-recall, micro-F1 y macro-F1 global frente a `weighted_t14t16`, pero baja F1 en T12, T14 y T16. Si la tesis prioriza recuperacion balanceada de toms, no reemplazar automaticamente la baseline oficial sin discutir ese trade-off.

## 4. Paquete minimo para inferencia sobre WAV reales

Para inferir sobre WAV reales con una run ya calibrada, conservar como minimo:

| Tipo | Archivo | Motivo |
| --- | --- | --- |
| Pesos seleccionados | `checkpoints/best_weights.weights.h5` o `checkpoints/final_weights.weights.h5`, segun `fulltrack_selected_inference_config.json` | Modelo efectivo para inferencia. |
| Metadata de checkpoint | `checkpoints/checkpoint_metadata.json` | Verifica esquema `vibro_5_toms`, labels y pesos/clases esperadas. |
| Config de entrenamiento | `config.json` | Reconstruye `model_name`, `sample_rate`, `context`, `training_sequence`, clase/loss y parametros relevantes. |
| Config calibrada | `fulltrack_calibration_*/fulltrack_selected_inference_config.json` | Umbral, NMS, tolerancia, checkpoint seleccionado y metrica TEST reportada. |
| Registro oficial, si aplica | `RUNS/current_official_baseline.json` y `.md` | Declara cual run usar como baseline y sus parametros recomendados. |

Para la baseline oficial actual, el paquete minimo es:

- `RUNS/serious_5class_low_60ep_weighted_t14t16/checkpoints/best_weights.weights.h5`
- `RUNS/serious_5class_low_60ep_weighted_t14t16/checkpoints/checkpoint_metadata.json`
- `RUNS/serious_5class_low_60ep_weighted_t14t16/config.json`
- `RUNS/serious_5class_low_60ep_weighted_t14t16/fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json`
- `RUNS/current_official_baseline.json`
- `RUNS/current_official_baseline.md`

Para la candidata global `focal_soft_toms_v2`, el paquete minimo equivalente es:

- `RUNS/serious_5class_low_60ep_focal_soft_toms_v2/checkpoints/best_weights.weights.h5`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v2/checkpoints/checkpoint_metadata.json`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v2/config.json`
- `RUNS/serious_5class_low_60ep_focal_soft_toms_v2/fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json`

## 5. Archivos esenciales para trazabilidad experimental

| Familia | Archivos | Conservar |
| --- | --- | --- |
| Entrenamiento | `config.json`, `history.csv`, `history.json`, `README.md` | Si. Permiten reproducir contexto y curva de entrenamiento. |
| Dataset usado | `dataset_event_counts.csv`, `dataset_event_counts.json` | Si. Evidencia balance por clase/split usado por la run. |
| Checkpoints | `checkpoints/checkpoint_metadata.json`, pesos `best/final` | Si para runs candidatas/actuales. Para historicas, REVISAR segun espacio. |
| Seleccion de checkpoint/threshold | `checkpoint_selection_from_val.json`, `selected_thresholds_from_val.json`, `*_metrics_by_threshold.*` | Si para runs que fundamentan resultados. |
| Calibracion full-track | `fulltrack_calibration_summary.md`, `fulltrack_nms_summary.md`, `fulltrack_selected_inference_config.json`, `fulltrack_nms_selected_config_from_val.json`, `fulltrack_metric_consistency_check.json`, CSV de VAL/TEST | Si para baseline/candidatas. |
| Analisis de errores | `error_summary.md`, `error_analysis_config.json`, `metric_consistency_check.json`, `ground_truth_count_check.json`, `error_analysis_by_class/profile/track.csv`, `worst_tracks_*.csv`, `confusion_candidates.csv` | Si si se cita en tesis; si no, REVISAR. |
| Comparaciones | `comparison_summary.json`, `comparison_summary.md`, `aggregate_metrics_comparison.csv`, `class_metrics_comparison.csv`, `README.md` | Si para justificar baseline/candidata. |
| Reportes renderizados | `*.pdf` | Secundarios si existe JSON/CSV/MD fuente. REVISAR, no eliminar automaticamente. |

## 6. Archivos pesados o secundarios observados

Los checkpoints no son el problema principal de espacio: cada `*.weights.h5` observado pesa aprox. 1.545 MB.

El volumen pesado aparece en analisis de errores repetidos, especialmente en `serious_5class_low_60ep_calibrated`:

| Carpeta | Tamano aprox. | Archivo mas pesado | Tamano aprox. | Clasificacion |
| --- | ---: | --- | ---: | --- |
| `error_analysis` | 33.773 MB | `event_level_errors.csv` | 30.864 MB | REVISAR. Detalle granular pesado. |
| `error_analysis_countcheck_full_both` | 33.773 MB | `event_level_errors.csv` | 30.864 MB | REVISAR. Parece solaparse con `error_analysis`. |
| `error_analysis_gtfix_full_both` | 24.206 MB | `event_level_errors.csv` | 20.844 MB | REVISAR. |
| `error_analysis_countcheck_full_global` | 16.844 MB | `event_level_errors.csv` | 15.395 MB | REVISAR. |
| `error_analysis_gtfix_full_global` | 12.236 MB | `event_level_errors.csv` | 10.395 MB | REVISAR. |

Clasificacion general:

- `event_level_errors.csv`: muy util para debugging fino, pero pesado y repetible; conservar solo si se cita o se necesita inspeccion granular. Marcar `REVISAR`.
- `error_analysis_by_track.csv`, `by_class.csv`, `by_profile.csv`, `worst_tracks_*.csv`: mucho mas compactos y de alto valor interpretativo; conservar.
- PDFs de reportes: utiles para lectura, pero secundarios si los datos fuente existen; `REVISAR`.
- Grillas completas `fulltrack_nms_*metrics.csv` y `fulltrack_*metrics_by_threshold.csv`: pequenas en estas runs, conviene conservar para trazabilidad.
- `final_weights.weights.h5` cuando la config selecciona `best`: no es necesario para inferir con la config seleccionada, pero puede ser necesario para reproducir comparaciones best vs final. Marcar `REVISAR`, no eliminar.

## 7. Smoke runs recientes

Clasificacion liviana, sin auditoria exhaustiva:

| Run smoke | Tamano aprox. | Checkpoints | Clasificacion |
| --- | ---: | ---: | --- |
| `smoke_without_wrappers` | 0.005 MB | 0 | Descartable/reproducible; REVISAR antes de borrar. |
| `smoke_wrapper_train` | 0.005 MB | 0 | Descartable/reproducible; REVISAR antes de borrar. |
| `PRUEBA_DESPUES_DE_AUDITORIA` | 0.005 MB | 0 | Descartable/reproducible; REVISAR antes de borrar. |
| `smoke_focal_loss_1ep` | 3.095 MB | 2 | Smoke con checkpoints; descartable si no se cita, pero REVISAR. |
| `smoke_class_weights_1ep` | 3.129 MB | 2 | Smoke con checkpoints; descartable si no se cita, pero REVISAR. |
| `smoke_calibrated_*` | 3.1-3.3 MB c/u | 2 c/u | Smoke historicos; REVISAR en fase de limpieza. |
| `smoke_custom_5class_tomboost_*` | ~0.003-0.004 MB | 0 | Descartables/reproducibles; REVISAR. |
| `_smoke_tmp` | 0.283 MB | 0 | Temporal; REVISAR. |

## 8. Conviene ajustar scripts?

Sin modificar scripts en esta fase, recomendaciones:

1. `SCRIPTS/analysis/analyze_run_errors.py`
   - Conviene agregar en una fase posterior un modo liviano, por ejemplo `--skip-event-level-errors` o `--summary-only`.
   - Motivo: actualmente siempre escribe `event_level_errors.csv`, que domina el peso de las carpetas de analisis.
   - Tambien convendria un `--top-k-worst` configurable para limitar salidas cuando solo se quiere diagnostico rapido.

2. `SCRIPTS/calibration/calibrate_fulltrack_thresholds.py`
   - No parece urgente reducir outputs: las grillas full-track/NMS son pequenas y trazables.
   - Mantener siempre `fulltrack_selected_inference_config.json`; es el archivo clave para WAV reales.

3. `SCRIPTS/training/train_custom_split.py`
   - Los outputs son razonables: config, history, counts, checkpoint metadata y pesos best/final.
   - Si se requiere ahorrar espacio en una fase posterior, revisar si conservar ambos checkpoints en runs no oficiales. Para baseline/candidatas actuales, mantener.

4. `SCRIPTS/analysis/compare_fulltrack_runs.py`
   - Outputs pequenos y utiles. No requiere ajuste inmediato.

## 9. Observaciones REVISAR

- `fulltrack_selected_inference_config.json` y `config.json` guardan `dataset_root` apuntando a `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` en algunas runs, aunque el dataset vigente actual esta en `DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`. Mantener historico, pero documentar la equivalencia o migracion antes de reproducir.
- `weighted_balanced_v2_vs_weighted_t14t16` tiene una posible inconsistencia nominal: la carpeta de comparacion dice `v2`, el JSON de comparacion apunta a candidata `serious_5class_low_60ep_weighted_balanced_v3`, y la inference config observada esta en `serious_5class_low_60ep_weighted_balanced_v2`. No borrar nada; revisar manualmente antes de usar esa comparacion.
- `serious_5class_low_60ep_weighted_balanced_v2` contiene calibracion full-track pero no checkpoints observados en esa carpeta. Puede depender de otra run o ser output parcial. REVISAR.
- `serious_5class_low_60ep_weighted_balanced_v3` contiene checkpoints pero no inference config full-track observada. REVISAR antes de inferir.
- `serious_5class_low_60ep_weighted_sd_light_keep_toms_v2` aparece en comparaciones pero no se observo `fulltrack_selected_inference_config.json` en la carpeta. REVISAR.

## 10. Comandos ejecutados

```powershell
git status --short
```

Resumen: workspace con muchos cambios previos, incluyendo `RUNS/` no trackeado. No se modificaron resultados.

```powershell
Get-ChildItem .\RUNS -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object Name, LastWriteTime
```

Resumen: las runs recientes principales son `serious_5class_low_60ep_focal_soft_toms_v2`, `serious_5class_low_60ep_focal_toms_v1`, candidatas weighted de julio/junio, `COMPARISONS` y smoke runs del 2026-07-13.

```powershell
rg --files .\RUNS\<runs seleccionadas>
rg --files .\RUNS\COMPARISONS
```

Resumen: se inventariaron archivos de runs baseline/candidatas y comparaciones asociadas sin leer binarios pesados.

```powershell
Get-Content .\RUNS\current_official_baseline.json
Get-Content .\RUNS\current_official_baseline.md
```

Resumen: baseline oficial actual es `serious_5class_low_60ep_weighted_t14t16`, checkpoint `best`, threshold global `0.03`, NMS `110 ms`.

```powershell
Get-Content .\RUNS\COMPARISONS\*\comparison_summary.json
Get-Content .\RUNS\COMPARISONS\*\comparison_summary.md
```

Resumen: se extrajeron deltas de micro-F1 y trade-offs por clase.

```powershell
Get-ChildItem .\RUNS\<runs seleccionadas> -Recurse -File
```

Resumen: se estimaron conteos/tamanos por run y se identificaron outputs pesados, sin borrar ni mover nada.

Confirmacion de esta fase:

- Solo se creo documentacion Markdown: `SCRIPTS/RUNS_OUTPUT_AUDIT.md`.
- No se modificaron scripts `.py`.
- No se borraron, movieron ni renombraron archivos.
- No se tocaron datasets, checkpoints, resultados, `RUNS/` ni outputs pesados.
- No se hizo commit ni push.
