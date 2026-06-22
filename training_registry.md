# Registro de pruebas y entrenamiento

Fecha de corte: 2026-06-21.

## Esquema de entrenamiento actual

El entrenamiento usa el escenario `custom_split` agregado al DataLoader de ADTOF. Este escenario busca datasets con splits fijos:

```text
DATASET_ROOT/
  TRAIN/AUDIO
  TRAIN/ANNOTATIONS
  VAL/AUDIO
  VAL/ANNOTATIONS
  TEST/AUDIO
  TEST/ANNOTATIONS
```

El vocabulario activo para vibraciones es:

```text
class_schema = vibro_5_toms
class_names  = KD, SD, T12, T14, T16
labels       = 35, 38, 47, 45, 43
weights      = 1.0, 1.0, 1.0, 1.0, 1.0
```

El script de entrenamiento principal es:

`SCRIPTS/mini_train_custom_split.py`

Aunque el nombre dice `mini_train`, actualmente sirve para smoke training, entrenamientos de 10 epochs, guardado de checkpoints y evaluacion en TEST.

## Pruebas unitarias y smoke tests

### Conversion JSON a ADTOF TXT

Script:

`SCRIPTS/test_json_to_adtof_txt.py`

Valida que:

- El vocabulario vibratorio sea independiente del vocabulario original reducido de ADTOF.
- T12, T14 y T16 se escriban como clases MIDI separadas.
- `MIDI_VIBRO_5` preserve los pitches esperados.

Comando:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_json_to_adtof_txt.py
```

### Cuotas tomboost

Script:

`SCRIPTS/test_tomboost_dataset_generator.py`

Valida que:

- La asignacion de cuotas incluya eventos positivos para todas las clases.
- KD y SD queden alrededor de 30% a 35%.
- Cada tom quede alrededor de 10% a 15%.
- El mapeo final sea KD=35, SD=38, T12=47, T14=45, T16=43.

Comando:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_tomboost_dataset_generator.py
```

### Smoke test del DataLoader custom_split

Script:

`SCRIPTS/test_custom_dataset_loader.py`

Valida:

- Pares WAV/TXT por split.
- Copia temporal de un subconjunto pequeno.
- Creacion de cache de features `.npy`.
- Construccion de `Track`, `x`, `yDense` y `sampleWeight`.
- Batch de entrenamiento con forma esperada.
- Uso del vocabulario vibratorio de 5 clases.

Comando recomendado para el dataset full:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --tracks-per-split 2 `
  --training-sequence 64 `
  --batch-size 1 `
  --sample-rate 100 `
  --context 9
```

Para conservar el subconjunto temporal:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --tracks-per-split 2 `
  --keep-subset `
  --work-dir .\RUNS\_smoke_tmp
```

## Entrenamientos realizados

### CUSTOM_DEBUG inicial

Carpeta relacionada:

`CUSTOM_DEBUG`

Corridas relacionadas:

- `RUNS/debug_10ep`
- `RUNS/debug_10ep_th03`
- `RUNS/debug_10ep_multi_threshold`
- `RUNS/debug_10ep_onset_eval`
- `RUNS/debug_eval_quick`

Objetivo:

Probar que el pipeline ADTOF podia entrenar usando `custom_split`, cachear features, producir batches y evaluar metricas sin depender de los datasets originales.

Resultado:

El pipeline de entrenamiento quedo validado de forma incremental. Estas corridas son historicas y menos representativas que las corridas 5-class actuales.

### Smoke 5-class tomboost

Corridas relacionadas:

- `RUNS/smoke_custom_5class_tomboost_20260615_1step`
- `RUNS/smoke_custom_5class_tomboost_20260615_1step2`
- `RUNS/smoke_custom_5class_tomboost_20260615_1step3`
- `RUNS/smoke_custom_5class_tomboost_20260615_1step5CLASS`
- `RUNS/smoke_custom_5class_tomboost_20260615_1step5CLASS2DO`
- `RUNS/smoke_custom_5class_tomboost_20260615_1stepNUEVO`

Objetivo:

Confirmar que el entrenamiento funcionaba con el nuevo esquema `vibro_5_toms` y que los checkpoints quedaban marcados con metadata de clase para evitar cargar pesos incompatibles del esquema original.

Configuracion representativa de `smoke_custom_5class_tomboost_20260615_1stepNUEVO`:

- Dataset: `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_DEBUG`
- Modelo: `Frame_RNN`
- Epochs: 10
- Steps per epoch: 100
- Validation steps: 1
- Batch size: 1
- Training sequence: 64
- Context: 9
- Sample rate: 100
- Evaluacion onset: desactivada en esta corrida.

### Debug multiprofile 5-class

Carpeta de corrida:

`RUNS/debug_5class_multiprofile_eval`

Dataset:

`PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_DEBUG`

Configuracion:

- Epochs: 10
- Steps per epoch: 100
- Validation steps: 20
- Batch size: 1
- Training sequence: 64
- Context: 9
- Sample rate: 100
- Evaluacion TEST: activada.
- Evaluacion onset: activada.
- Tolerancia onset: 50 ms.
- Umbrales: 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5.

Historia de perdida:

- Epoch 1: loss 0.1326, val_loss 0.2515.
- Epoch 6: loss 0.0610, val_loss 0.0514.
- Epoch 10: loss 0.0812, val_loss 0.0549.

Lectura:

El modelo entreno correctamente y produjo metricas onset. Los resultados confirmaron que el pipeline funcionaba, pero tambien mostraron sensibilidad fuerte al umbral.

### Full 5-class low 10ep eval

Carpeta de corrida:

`RUNS/full_5class_low_10ep_eval`

Dataset:

`PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`

Configuracion:

- Epochs: 10
- Steps per epoch: 200
- Validation steps: 50
- Batch size: 1
- Training sequence: 64
- Context: 9
- Sample rate: 100
- Same padding: false.
- Evaluacion TEST: activada.
- Evaluacion onset: activada.
- Tolerancia onset: 50 ms.
- Checkpoints: activados.

Archivos generados importantes:

- `config.json`
- `history.csv`
- `history.json`
- `dataset_event_counts.csv`
- `dataset_event_counts.json`
- `final_test_metrics_by_threshold.csv`
- `final_onset_metrics_by_threshold.csv`
- `final_prediction_stats.csv`
- `checkpoints/checkpoint_metadata.json`
- `checkpoints/best_weights.weights.h5`
- `checkpoints/final_weights.weights.h5`

Historia de perdida:

```text
epoch  loss      val_loss
1      0.0969    4.0906
2      0.0780    1.6625
3      0.0737    0.5352
4      0.0684    0.0731
5      0.0583    0.0693
6      0.0693    0.0723
7      0.0629    0.0660
8      0.0656    0.0696
9      0.0616    0.0631
10     0.0637    0.0645
```

Lectura:

La perdida se estabilizo despues de las primeras epochs. El salto inicial alto en `val_loss` bajo rapidamente, lo que sugiere que el entrenamiento si esta aprendiendo el formato de targets.

Mejores senales onset observadas en la corrida full:

- KD alcanza F1 onset cercano a 0.435 en threshold 0.1.
- SD alcanza F1 onset cercano a 0.464 en threshold 0.05.
- T12 alcanza F1 onset cercano a 0.459 en threshold 0.1.
- T16 alcanza F1 onset cercano a 0.442 en threshold 0.15.
- T14 es la clase mas debil; cae rapido al subir el threshold y requiere revision.

Conclusion actual:

El entrenamiento full es funcional y reproducible, pero todavia no debe considerarse modelo final. La siguiente etapa debe enfocarse en mas entrenamiento, calibracion de thresholds por clase y revision especifica de T14.

## Comandos de entrenamiento

### Smoke training minimo

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --run-name smoke_full_1ep `
  --epochs 1 `
  --steps-per-epoch 2 `
  --validation-steps 1 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --output-dir .\RUNS
```

### Entrenamiento debug con evaluacion

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG `
  --run-name debug_low_density_10ep_eval `
  --epochs 10 `
  --steps-per-epoch 100 `
  --validation-steps 20 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --save-checkpoints `
  --eval-test `
  --eval-steps 20 `
  --eval-onsets `
  --onset-tolerance-ms 50 `
  --output-dir .\RUNS
```

### Entrenamiento full reproducible

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --run-name full_5class_low_10ep_eval_v2 `
  --epochs 10 `
  --steps-per-epoch 200 `
  --validation-steps 50 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --save-checkpoints `
  --eval-test `
  --eval-steps 20 `
  --eval-onsets `
  --onset-tolerance-ms 50 `
  --thresholds 0.03 0.05 0.075 0.1 0.15 0.2 0.3 0.4 0.5 `
  --output-dir .\RUNS
```

### Entrenamiento mas largo sugerido

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --run-name full_5class_low_30ep_eval `
  --epochs 30 `
  --steps-per-epoch 400 `
  --validation-steps 100 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --save-checkpoints `
  --eval-test `
  --eval-steps 50 `
  --eval-onsets `
  --onset-tolerance-ms 50 `
  --thresholds 0.02 0.03 0.05 0.075 0.1 0.125 0.15 0.2 0.3 0.4 0.5 `
  --output-dir .\RUNS
```

## Diagnosticos antes de entrenar

Antes de correr un entrenamiento largo:

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_wav_dataset_format.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --peak-limit 0.9

conda run -n adtof_rod python .\SCRIPTS\check_dataset_temporal_consistency.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL

conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --tracks-per-split 2
```

## Notas de seguridad de checkpoints

`mini_train_custom_split.py` guarda `checkpoint_metadata.json` junto a los pesos. Esto existe para evitar cargar checkpoints incompatibles del esquema original de ADTOF, que usaba clases reducidas distintas. Si un checkpoint no tiene metadata o no coincide con `vibro_5_toms`, el script debe rechazarlo.

## Estado pendiente

- Calibrar thresholds por clase en vez de usar 0.5 fijo.
- Investigar T14: revisar muestras fuente, mezcla, balance real y separabilidad respecto a T12/T16.
- Probar entrenamientos mas largos y con mas `steps_per_epoch`.
- Guardar un resumen humano por corrida en `RUNS/<run_name>/README.md` para comparar rapidamente.
