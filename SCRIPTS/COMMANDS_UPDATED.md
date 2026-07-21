# Comandos actualizados despues de reorganizar SCRIPTS

Fecha: 2026-07-13

Alcance: comandos probables de uso real despues de mover scripts a subcarpetas funcionales. Esta fase mantiene wrappers temporales en la raiz de `SCRIPTS/`; no se borran scripts funcionales ni se tocan datasets, `RUNS`, checkpoints u outputs pesados.

Contexto vigente:

- Proyecto ADTOF adaptado a senales vibratorias/piezos.
- Esquema `vibro_5_toms`: KD=35, SD=38, T12=47, T14=45, T16=43.
- Metrica principal: onset-based F1 con tolerancia temporal.
- Generador vigente: `DATASET_GENERATOR`.
- Dataset principal: `DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`.

## Rutas nuevas

| Uso | Ruta nueva |
| --- | ---------- |
| Check WAV | `SCRIPTS/dataset_checks/check_wav_dataset_format.py` |
| Check temporal WAV/JSON/TXT | `SCRIPTS/dataset_checks/check_dataset_temporal_consistency.py` |
| Conteo de eventos | `SCRIPTS/dataset_checks/analyze_dataset_event_counts.py` |
| Smoke DataLoader custom_split | `SCRIPTS/smoke_tests/smoke_custom_dataset_loader.py` |
| Smoke JSON a ADTOF txt | `SCRIPTS/smoke_tests/smoke_json_to_adtof_txt.py` |
| Smoke generador tomboost/multiperfil | `SCRIPTS/smoke_tests/smoke_tomboost_dataset_generator.py` |
| Entrenamiento controlado | `SCRIPTS/training/train_custom_split.py` |
| Calibracion full-track | `SCRIPTS/calibration/calibrate_fulltrack_thresholds.py` |
| Analisis de errores por run | `SCRIPTS/analysis/analyze_run_errors.py` |
| Comparacion de runs full-track | `SCRIPTS/analysis/compare_fulltrack_runs.py` |

## Validacion de dataset

```powershell
$datasetRoot = ".\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL"
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\dataset_checks\check_wav_dataset_format.py --dataset-root $datasetRoot
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\dataset_checks\check_dataset_temporal_consistency.py --dataset-root $datasetRoot
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\dataset_checks\analyze_dataset_event_counts.py --counts-csv "$datasetRoot\dataset_event_counts.csv"
```

## Smoke tests

```powershell
conda run -n adtof_rod python .\SCRIPTS\smoke_tests\smoke_json_to_adtof_txt.py
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\smoke_tests\smoke_tomboost_dataset_generator.py
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\smoke_tests\smoke_custom_dataset_loader.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --tracks-per-split 2 `
  --training-sequence 64 `
  --batch-size 1 `
  --sample-rate 100 `
  --context 9
```

## Entrenamiento smoke minimo

```powershell
conda run -n adtof_rod python .\SCRIPTS\training\train_custom_split.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --run-name smoke_after_scripts_reorg `
  --epochs 1 `
  --steps-per-epoch 2 `
  --validation-steps 1 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --output-dir .\RUNS
```

## Calibracion y analisis de runs

Usar un `run-dir` existente que ya tenga checkpoint y metadata compatibles con `vibro_5_toms`.

```powershell
conda run -n adtof_rod python .\SCRIPTS\calibration\calibrate_fulltrack_thresholds.py `
  --run-dir .\RUNS\<RUN_NAME> `
  --thresholds 0.03 0.05 0.075 0.1 0.15 0.2 0.3 0.4 0.5 `
  --output-dir .\RUNS\<RUN_NAME>\fulltrack_calibration `
  --overwrite
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\analysis\analyze_run_errors.py `
  --run-dir .\RUNS\<RUN_NAME> `
  --split TEST `
  --output-dir .\RUNS\<RUN_NAME>\error_analysis `
  --overwrite
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\analysis\compare_fulltrack_runs.py `
  --baseline-run .\RUNS\<BASELINE_RUN> `
  --candidate-run .\RUNS\<CANDIDATE_RUN> `
  --baseline-config .\RUNS\<BASELINE_RUN>\fulltrack_calibration\fulltrack_selected_inference_config.json `
  --candidate-config .\RUNS\<CANDIDATE_RUN>\fulltrack_calibration\fulltrack_selected_inference_config.json `
  --output-dir .\RUNS\COMPARISONS\<COMPARISON_NAME> `
  --overwrite
```

## Wrappers temporales conservados

Estos comandos antiguos siguen funcionando durante la transicion:

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py --help
conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py --help
conda run -n adtof_rod python .\SCRIPTS\test_json_to_adtof_txt.py
conda run -n adtof_rod python .\SCRIPTS\test_tomboost_dataset_generator.py
conda run -n adtof_rod python .\SCRIPTS\check_wav_dataset_format.py --help
conda run -n adtof_rod python .\SCRIPTS\check_dataset_temporal_consistency.py --help
conda run -n adtof_rod python .\SCRIPTS\analyze_dataset_event_counts.py --help
conda run -n adtof_rod python .\SCRIPTS\calibrate_fulltrack_thresholds.py --help
conda run -n adtof_rod python .\SCRIPTS\analyze_run_errors.py --help
conda run -n adtof_rod python .\SCRIPTS\compare_fulltrack_runs.py --help
```

## Como verificar si los wrappers pueden eliminarse

No eliminar wrappers en esta fase. En una fase posterior, primero confirmar que no quedan referencias operativas a rutas antiguas fuera de auditorias historicas:

```powershell
rg -n --hidden `
  --glob "!RUNS/**" `
  --glob "!OUTPUT_TRANSCRIPTIONS/**" `
  --glob "!DATASET_GENERATOR/**/AUDIO/**" `
  --glob "!DATASET_GENERATOR/**/LABELS/**" `
  --glob "!DATASET_GENERATOR/**/ANNOTATIONS/**" `
  "SCRIPTS[\\/](mini_train_custom_split|test_custom_dataset_loader|test_json_to_adtof_txt|test_tomboost_dataset_generator|check_wav_dataset_format|check_dataset_temporal_consistency|analyze_dataset_event_counts|calibrate_fulltrack_thresholds|analyze_run_errors|compare_fulltrack_runs)\.py" .
```

Despues revisar manualmente cada coincidencia:

- Mantener coincidencias dentro de auditorias historicas, por ejemplo `SCRIPTS/AUDIT_SCRIPTS.md`, `SCRIPTS/AUDIT_SCRIPTS_PHASE2A_REFERENCES.md` y `SCRIPTS/COMMAND_USAGE_AUDIT.md`.
- Actualizar coincidencias en documentacion vigente, scripts auxiliares o comandos de uso diario.
- Confirmar que no hay imports funcionales a wrappers:

```powershell
rg -n "from SCRIPTS\.(mini_train_custom_split|analyze_run_errors)|import SCRIPTS\.(mini_train_custom_split|analyze_run_errors)" .\SCRIPTS --glob "*.py"
```

Si las unicas referencias restantes son historicas y los comandos nuevos fueron validados, los wrappers pueden retirarse en una fase separada con revision explicita.
