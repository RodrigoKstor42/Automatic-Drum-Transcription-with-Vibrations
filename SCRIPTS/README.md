# SCRIPTS

## Propósito

Esta carpeta contiene scripts para diagnóstico de dataset, smoke tests, entrenamiento, calibración, evaluación, análisis de errores y comparación de runs del proyecto ADTOF adaptado a señales vibratorias/piezos.

## Scripts vigentes

La organizacion vigente agrupa los scripts en subcarpetas: `dataset_checks/`, `smoke_tests/`, `training/`, `calibration/`, `analysis/` y `environment_patches/`. Los nombres historicos en la raiz de `SCRIPTS/` pueden seguir existiendo como wrappers de compatibilidad, pero para reproducir el proyecto conviene usar los comandos de `SCRIPTS/COMMANDS_UPDATED.md`.

| Script | Categoría | Uso | Riesgo/nota |
|---|---|---|---|
| `check_wav_dataset_format.py` | diagnóstico de dataset | Valida formato técnico de WAV, sample rate, canales, subtipo y clipping. | Lectura de dataset; puede tardar en datasets grandes. |
| `check_dataset_temporal_consistency.py` | diagnóstico de dataset | Revisa duraciones WAV/anotaciones y eventos fuera de rango. | Lectura completa del dataset. |
| `analyze_dataset_event_counts.py` | diagnóstico de dataset | Analiza balance de eventos por split y clase. | Puede escribir resúmenes si se usa `--run-dir`. |
| `test_custom_dataset_loader.py` | smoke test | Verifica carga `custom_split`, labels, cache y batch del DataLoader. | Crea subconjunto temporal; requiere `--dataset-root`. |
| `test_json_to_adtof_txt.py` | smoke test | Verifica conversión JSON a `.txt` ADTOF y vocabulario vigente. | Usa directorios temporales. |
| `test_tomboost_dataset_generator.py` | smoke test | Verifica cuotas, perfiles y mapeo MIDI del generador. | No debe tocar datasets principales. |
| `mini_train_custom_split.py` | entrenamiento | Ejecuta entrenamientos controlados sobre `custom_split`, con evaluación opcional. | Escribe en `RUNS`; puede crear checkpoints si se pide. |
| `calibrate_fulltrack_thresholds.py` | calibración/evaluación | Calibra thresholds/NMS desde VAL y reporta TEST. | Carga checkpoints y escribe outputs de calibración. |
| `analyze_run_errors.py` | análisis de errores | Reconstruye TP/FP/FN por track, clase y perfil. | Depende de run/checkpoint/config válidos. |
| `compare_fulltrack_runs.py` | comparación de runs | Compara baseline y candidato con métricas full-track. | `--overwrite` puede reemplazar el directorio de salida. |

## Orden recomendado de uso

1. Generar dataset en `DATASET_GENERATOR`.
2. Validar WAV.
3. Validar consistencia temporal.
4. Analizar conteo de eventos.
5. Probar DataLoader.
6. Smoke training.
7. Entrenamiento real.
8. Calibrar thresholds/NMS.
9. Analizar errores.
10. Comparar runs.

Comandos PowerShell de referencia:

Nota: esta seccion conserva comandos historicos de compatibilidad. Para ejecucion diaria y reproduccion final, usar `SCRIPTS/COMMANDS_UPDATED.md`, que ya apunta a las subcarpetas vigentes.

```powershell
$datasetRoot = ".\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL"
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_wav_dataset_format.py --dataset-root $datasetRoot
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_dataset_temporal_consistency.py --dataset-root $datasetRoot
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\analyze_dataset_event_counts.py --counts-csv "$datasetRoot\dataset_event_counts.csv"
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py `
  --dataset-root $datasetRoot `
  --tracks-per-split 2 `
  --training-sequence 64 `
  --batch-size 1 `
  --sample-rate 100 `
  --context 9
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_json_to_adtof_txt.py
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_tomboost_dataset_generator.py
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py `
  --dataset-root $datasetRoot `
  --run-name smoke_5class_low_1ep `
  --epochs 1 `
  --steps-per-epoch 2 `
  --validation-steps 1 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --output-dir .\RUNS `
  --eval-test `
  --eval-onsets
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\calibrate_fulltrack_thresholds.py `
  --run-dir .\RUNS\NOMBRE_RUN `
  --thresholds 0.03 0.05 0.075 0.1 0.15 `
  --nms-grid-ms 0 20 40 60 `
  --output-dir .\RUNS\NOMBRE_RUN\fulltrack_calibration_f1_nms
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\analyze_run_errors.py `
  --run-dir .\RUNS\NOMBRE_RUN `
  --split TEST `
  --threshold-mode both `
  --output-dir .\RUNS\NOMBRE_RUN\error_analysis
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\compare_fulltrack_runs.py `
  --baseline-run .\RUNS\BASELINE `
  --candidate-run .\RUNS\CANDIDATO `
  --baseline-config .\RUNS\BASELINE\fulltrack_selected_inference_config.json `
  --candidate-config .\RUNS\CANDIDATO\fulltrack_selected_inference_config.json `
  --output-dir .\RUNS\COMPARISONS\BASELINE_vs_CANDIDATO
```

## Nota sobre nombres

`mini_train_custom_split.py` se mantiene por compatibilidad, pero actualmente sirve para entrenamientos controlados serios.

Los scripts `test_*` son smoke tests vigentes, no scripts obsoletos.

`fixmadmom.py` fue movido a `environment_patches/` porque modifica el entorno local.

## Métrica principal

La métrica principal del proyecto es onset-based F1 con tolerancia temporal. No usar frame-wise F1 como métrica principal para decidir modelos.

## Esquema vigente

| Clase | MIDI |
|---|---:|
| KD | 35 |
| SD | 38 |
| T12 | 47 |
| T14 | 45 |
| T16 | 43 |
