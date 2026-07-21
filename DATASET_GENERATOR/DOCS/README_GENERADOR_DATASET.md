# Guia extendida del generador de datasets

## Vision general del pipeline

`DATASET_GENERATOR` genera frases procedurales de bateria usando muestras vibratorias individuales almacenadas en `DATASET_GENERATOR/MUESTRAS/`. El resultado esperado es un dataset entrenable para ADTOF con splits `TRAIN`, `VAL` y `TEST`.

Cada split contiene:

```text
AUDIO/
LABELS/
ANNOTATIONS/
```

El esquema vigente es `vibro_5_toms`:

- KD = 35
- SD = 38
- T12 = 47
- T14 = 45
- T16 = 43

El esquema antiguo KD/SD/TT/HH/CY no corresponde al pipeline vigente.

## Componentes principales

- `SRC/phrase_generator.py`: implementacion principal de generacion.
- `SRC/generation_profiles.py`: perfiles musicales reutilizables.
- `generate_tomboost_dataset.py`: wrapper de compatibilidad para ejecutar el generador.
- `json_to_adtof_txt.py`: conversion manual de JSON a anotaciones `.txt`.
- `MUESTRAS/`: muestras fuente locales, no versionadas.
- `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL/`: dataset principal actual, local.
- `DATASETS_DE_PRUEBA/`: datasets historicos o debug.
- `legacy/`: scripts historicos.

## Perfiles de generacion

Los perfiles actuales estan definidos en `SRC/generation_profiles.py`:

- `basic_groove`
- `standard_rock`
- `fill_heavy`
- `fast_dense`
- `humanized_mixed`

Cada perfil controla rango de BPM, cantidad de eventos, probabilidad de fills, probabilidad de toms, densidad ritmica, overlaps naturales, humanizacion temporal y variacion dinamica.

## Parametros importantes

- `--output-root`: carpeta donde se escribira el dataset. Debe ser nueva o estar vacia.
- `--num-train`: cantidad de frases para `TRAIN`.
- `--num-val`: cantidad de frases para `VAL`.
- `--num-test`: cantidad de frases para `TEST`.
- `--profiles`: lista de perfiles a mezclar.
- `--profile-weights`: pesos relativos de los perfiles.
- `--tom-boost`: multiplica la presencia de toms/fills.
- `--target-tom-share`: proporcion objetivo total para T12, T14 y T16.
- `--balance-individual-toms`: balancea T12, T14 y T16 individualmente.
- `--peak-limit`: limite de peak para reducir clipping.
- `--write-annotations`: escribe anotaciones ADTOF `.txt` durante la generacion.
- `--seed`: semilla de reproducibilidad.

## Ejemplo smoke test

```powershell
conda run -n adtof_rod python .\DATASET_GENERATOR\generate_tomboost_dataset.py `
  --output-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --num-train 20 `
  --num-val 5 `
  --num-test 5 `
  --profiles basic_groove standard_rock fill_heavy fast_dense humanized_mixed `
  --profile-weights 0.3 0.3 0.15 0.1 0.15 `
  --tom-boost 1.3 `
  --target-tom-share 0.30 `
  --balance-individual-toms `
  --peak-limit 0.9 `
  --write-annotations `
  --seed 43
```

## Ejemplo dataset completo comparable

```powershell
conda run -n adtof_rod python .\DATASET_GENERATOR\generate_tomboost_dataset.py `
  --output-root .\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL_REGEN `
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

## Estructura esperada del output

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

## Conversion manual de anotaciones

Si no se uso `--write-annotations`, convertir JSON a `.txt` con:

```powershell
conda run -n adtof_rod python .\DATASET_GENERATOR\json_to_adtof_txt.py `
  --root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST
```

Formato:

```text
timestamp_seconds    midi_pitch    velocity
```

## Validacion posterior

Formato WAV:

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_wav_dataset_format.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --target-sr 12000 `
  --expected-subtype PCM_16 `
  --peak-limit 0.9
```

Consistencia temporal:

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_dataset_temporal_consistency.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --duration-tolerance-ms 5 `
  --min-tail-margin-ms 100
```

Conteo de eventos:

```powershell
conda run -n adtof_rod python .\SCRIPTS\analyze_dataset_event_counts.py `
  --counts-csv .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST\dataset_event_counts.csv
```

Smoke test del DataLoader:

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --tracks-per-split 2 `
  --training-sequence 64 `
  --batch-size 1 `
  --sample-rate 100 `
  --context 9
```

## Trazabilidad experimental

Para cada dataset generado, conservar:

- comando usado
- seed
- perfiles y pesos
- `dataset_config.json`
- `dataset_event_counts.csv`
- resultados de validacion
- ruta del dataset usado para entrenamiento

No sobrescribir datasets existentes. Usar un nombre nuevo por iteracion para mantener comparabilidad.
