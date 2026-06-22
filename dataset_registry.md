# Registro de datasets

Fecha de corte: 2026-06-21.

## Estructura esperada

Los datasets entrenables usan esta estructura:

```text
DATASET_ROOT/
  TRAIN/
    AUDIO/*.wav
    LABELS/*.json
    ANNOTATIONS/*.txt
  VAL/
    AUDIO/*.wav
    LABELS/*.json
    ANNOTATIONS/*.txt
  TEST/
    AUDIO/*.wav
    LABELS/*.json
    ANNOTATIONS/*.txt
  dataset_config.json
  dataset_event_counts.csv
  dataset_event_counts.json
```

El esquema de clases vigente es `vibro_5_toms`:

```text
KD=35, SD=38, T12=47, T14=45, T16=43
```

Los WAV deben estar en mono, 12 kHz y PCM16. Las anotaciones `.txt` usan filas separadas por tabulacion:

```text
timestamp_seconds    midi_pitch    velocity
```

## Iteracion 1: generacion raw normalizada

Carpetas relacionadas:

- `PHRASE_GENERATOR/GENERATED_RAW`
- `PHRASE_GENERATOR/CUSTOM_NORMALIZED`
- `PHRASE_GENERATOR/TRAIN`, `PHRASE_GENERATOR/VAL`, `PHRASE_GENERATOR/TEST`

Objetivo:

Crear las primeras frases procedurales sin distorsion fuerte, con golpes ordenados musicalmente y exportacion WAV/JSON. Esta etapa corrigio los problemas iniciales: distorsion, clipping, golpes superpuestos de forma poco natural y truncamiento de colas.

Comando base para generar frases por perfil:

```powershell
conda run -n drum_tools python .\PHRASE_GENERATOR\SRC\phrase_generator.py `
  --profile standard_rock `
  --count 1000 `
  --output-root .\PHRASE_GENERATOR\CUSTOM_NORMALIZED `
  --target-sr 12000 `
  --wav-subtype PCM_16 `
  --peak-limit 0.98
```

Comando para crear split desde un raw dataset:

```powershell
conda run -n drum_tools python .\PHRASE_GENERATOR\SRC\dataset_splitter.py `
  --raw-audio-dir .\PHRASE_GENERATOR\CUSTOM_NORMALIZED\AUDIO `
  --raw-labels-dir .\PHRASE_GENERATOR\CUSTOM_NORMALIZED\LABELS `
  --output-root .\PHRASE_GENERATOR `
  --summary-path .\PHRASE_GENERATOR\dataset_summary.json `
  --clean
```

Conversion JSON a `.txt` ADTOF:

```powershell
conda run -n drum_tools python .\PHRASE_GENERATOR\json_to_adtof_txt.py `
  --root .\PHRASE_GENERATOR
```

Motivo del cambio posterior:

El splitter funcionaba, pero convenia generar directamente `TRAIN`, `VAL`, `TEST`, `AUDIO`, `LABELS` y `ANNOTATIONS` desde un solo comando, con conteos de eventos y metadata de balance.

## Iteracion 2: CUSTOM_5CLASS_TOMBOOST

Carpeta:

`PHRASE_GENERATOR/CUSTOM_5CLASS_TOMBOOST`

Objetivo:

Crear un dataset de 5 clases con presencia explicita de los tres toms. La motivacion fue evitar que los toms quedaran subrepresentados si se dependia solo de frases tipo groove.

Configuracion registrada:

- TRAIN: 1000
- VAL: 200
- TEST: 200
- Perfil: `fill_heavy`
- `tom_boost`: 1.5
- `target_tom_share`: 0.36
- Toms balanceados individualmente.
- Target shares: KD 0.32, SD 0.32, T12 0.12, T14 0.12, T16 0.12.
- WAV: 12 kHz, mono, PCM16, peak limit 0.98.
- Seed: 7.

Comando reproducible:

```powershell
conda run -n adtof_rod python .\PHRASE_GENERATOR\generate_tomboost_dataset.py `
  --output-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_TOMBOOST `
  --num-train 1000 `
  --num-val 200 `
  --num-test 200 `
  --profile fill_heavy `
  --tom-boost 1.5 `
  --balance-individual-toms `
  --target-tom-share 0.36 `
  --write-annotations
```

Motivo del cambio posterior:

El dataset era util para probar el esquema 5 clases, pero estaba dominado por un solo perfil (`fill_heavy`). Se necesitaba mas variedad musical para que el modelo no aprendiera solo fills densos.

## Iteracion 3: CUSTOM_5CLASS_MULTIPROFILE_DEBUG

Carpeta:

`PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_DEBUG`

Objetivo:

Probar un dataset multiperfil con los cinco perfiles disponibles y mantener una cuota fuerte de toms.

Configuracion registrada:

- TRAIN: 2000
- VAL: 400
- TEST: 400
- Perfiles: `basic_groove`, `standard_rock`, `fill_heavy`, `fast_dense`, `humanized_mixed`.
- Pesos: 0.20, 0.20, 0.25, 0.15, 0.20.
- `tom_boost`: 1.8.
- `target_tom_share`: 0.36.
- Target shares: KD 0.32, SD 0.32, T12 0.12, T14 0.12, T16 0.12.
- WAV: 12 kHz, mono, PCM16, peak limit 0.98.
- Anotaciones `.txt`: activadas.
- Seed: 42.

Comando aproximado:

```powershell
conda run -n adtof_rod python .\PHRASE_GENERATOR\generate_tomboost_dataset.py `
  --output-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_DEBUG `
  --num-train 2000 `
  --num-val 400 `
  --num-test 400 `
  --profiles basic_groove standard_rock fill_heavy fast_dense humanized_mixed `
  --profile-weights 0.2 0.2 0.25 0.15 0.2 `
  --tom-boost 1.8 `
  --target-tom-share 0.36 `
  --balance-individual-toms `
  --peak-limit 0.98 `
  --write-annotations `
  --seed 42
```

Motivo del cambio posterior:

Las pruebas de entrenamiento mostraron que el modelo podia correr con el nuevo esquema, pero hacia falta una version menos densa y con menor peak limit para reducir saturacion y mejorar generalizacion.

## Iteracion 4: CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG

Carpeta:

`PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG`

Objetivo:

Probar una distribucion menos agresiva: mas peso en grooves basicos/rock, menor refuerzo de toms y peak limit mas conservador.

Configuracion registrada:

- TRAIN: 2000
- VAL: 400
- TEST: 400
- Perfiles: `basic_groove`, `standard_rock`, `fill_heavy`, `fast_dense`, `humanized_mixed`.
- Pesos: 0.30, 0.30, 0.15, 0.10, 0.15.
- `tom_boost`: 1.3.
- `target_tom_share`: 0.30.
- Target shares: KD 0.35, SD 0.35, T12 0.10, T14 0.10, T16 0.10.
- WAV: 12 kHz, mono, PCM16, peak limit 0.9.
- Anotaciones `.txt`: activadas.
- Seed: 43.

Comando reproducible:

```powershell
conda run -n adtof_rod python .\PHRASE_GENERATOR\generate_tomboost_dataset.py `
  --output-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG `
  --num-train 2000 `
  --num-val 400 `
  --num-test 400 `
  --profiles basic_groove standard_rock fill_heavy fast_dense humanized_mixed `
  --profile-weights 0.3 0.3 0.15 0.1 0.15 `
  --tom-boost 1.3 `
  --target-tom-share 0.30 `
  --balance-individual-toms `
  --peak-limit 0.9 `
  --write-annotations `
  --seed 43
```

Motivo del cambio posterior:

Esta configuracion paso a ser la base para una version completa de mayor tamano, manteniendo la misma logica de balance y control de peak.

## Iteracion 5: CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL

Carpeta:

`PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`

Estado:

Dataset principal actual para entrenamiento.

Configuracion registrada:

- TRAIN: 9600
- VAL: 1200
- TEST: 1200
- Total: 12000 frases.
- Perfiles: `basic_groove`, `standard_rock`, `fill_heavy`, `fast_dense`, `humanized_mixed`.
- Pesos: 0.30, 0.30, 0.15, 0.10, 0.15.
- `tom_boost`: 1.3.
- `target_tom_share`: 0.30.
- Target shares: KD 0.35, SD 0.35, T12 0.10, T14 0.10, T16 0.10.
- WAV: 12 kHz, mono, PCM16, peak limit 0.9.
- Anotaciones `.txt`: activadas.
- Seed: 43.

Conteo de eventos:

```text
TRAIN KD=39008 SD=39007 T12=11145 T14=11145 T16=11145
VAL   KD=4870  SD=4870  T12=1392  T14=1392  T16=1392
TEST  KD=4891  SD=4890  T12=1397  T14=1397  T16=1397
```

Comando reproducible:

```powershell
conda run -n adtof_rod python .\PHRASE_GENERATOR\generate_tomboost_dataset.py `
  --output-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --num-train 9600 `
  --num-val 1200 `
  --num-test 1200 `
  --profiles basic_groove standard_rock fill_heavy fast_dense humanized_mixed `
  --profile-weights 0.3 0.3 0.15 0.1 0.15 `
  --tom-boost 1.3 `
  --target-tom-share 0.30 `
  --balance-individual-toms `
  --peak-limit 0.9 `
  --write-annotations `
  --seed 43
```

Nota: el generador evita sobrescribir datasets con archivos existentes. Usar un nuevo `--output-root` si se quiere regenerar una version comparable.

## Diagnosticos de dataset

Formato WAV:

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_wav_dataset_format.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --target-sr 12000 `
  --expected-subtype PCM_16 `
  --peak-limit 0.9
```

Consistencia temporal WAV/JSON/TXT:

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_dataset_temporal_consistency.py `
  --dataset-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL `
  --duration-tolerance-ms 5 `
  --min-tail-margin-ms 100
```

Conteo y balance de eventos:

```powershell
conda run -n adtof_rod python .\SCRIPTS\analyze_dataset_event_counts.py `
  --counts-csv .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL\dataset_event_counts.csv
```

Si los conteos fueron guardados dentro de una carpeta de corrida:

```powershell
conda run -n adtof_rod python .\SCRIPTS\analyze_dataset_event_counts.py `
  --run-dir .\RUNS\full_5class_low_10ep_eval
```

Smoke test de conversion JSON a `.txt` y vocabulario 5 clases:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_json_to_adtof_txt.py
```

Smoke test de cuotas tomboost:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_tomboost_dataset_generator.py
```
