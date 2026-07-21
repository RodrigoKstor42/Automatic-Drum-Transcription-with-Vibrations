# DATASET_GENERATOR - Generador procedural de datasets vibratorios

## 1. Proposito

Esta carpeta contiene el generador procedural de frases de bateria para el proyecto ADTOF adaptado a senales vibratorias/piezos. Su objetivo es crear datasets entrenables con audios `.wav`, labels JSON y anotaciones ADTOF `.txt`.

El esquema vigente es `vibro_5_toms`:

- KD = 35
- SD = 38
- T12 = 47
- T14 = 45
- T16 = 43

El esquema antiguo KD/SD/TT/HH/CY no corresponde al pipeline vigente y solo debe aparecer como referencia historica.

## 2. Estructura general

```text
DATASET_GENERATOR/
  SRC/
    phrase_generator.py
    generation_profiles.py

  generate_tomboost_dataset.py
  json_to_adtof_txt.py

  MUESTRAS/
  CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL/
  DATASETS_DE_PRUEBA/
  legacy/
  DOCS/
  README.md
  requirements.yml
```

- `SRC/phrase_generator.py`: generador principal.
- `SRC/generation_profiles.py`: perfiles musicales de generacion.
- `generate_tomboost_dataset.py`: wrapper de compatibilidad para generar datasets.
- `json_to_adtof_txt.py`: conversor de labels JSON a anotaciones ADTOF `.txt`.
- `MUESTRAS/`: biblioteca local de golpes individuales, no versionada por Git.
- `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL/`: dataset principal actual, local/no versionado.
- `DATASETS_DE_PRUEBA/`: datasets historicos, debug o intermedios.
- `legacy/`: scripts historicos que ya no son parte del pipeline principal.
- `DOCS/`: documentacion extendida.

## 3. Descarga de muestras fuente

La carpeta `MUESTRAS/` es demasiado grande para GitHub y se distribuye por Google Drive:

```text
[PEGAR_AQUI_LINK_DE_GOOGLE_DRIVE_MUESTRAS]
```

Desde PowerShell, entra a la raiz del repositorio:

```powershell
cd "RUTA\AL\REPOSITORIO"
```

Descarga y descomprime/copia la biblioteca en:

```text
DATASET_GENERATOR/MUESTRAS/
```

Estructura esperada:

```text
DATASET_GENERATOR/
  MUESTRAS/
    KD/
      *.wav
    SD/
      *.wav
    T12/
      *.wav
    T14/
      *.wav
    T16/
      *.wav
```

Notas:

- No subir `MUESTRAS/` a Git.
- No renombrar clases sin actualizar el generador.
- Las muestras deben corresponder a KD, SD, T12, T14 y T16.

## 4. Entorno recomendado

El entorno usado actualmente en el proyecto suele ser `adtof_rod`:

```powershell
conda activate adtof_rod
```

Tambien puedes usar `conda run`:

```powershell
conda run -n adtof_rod python .\DATASET_GENERATOR\generate_tomboost_dataset.py --help
```

Si necesitas crear un entorno desde la especificacion local:

```powershell
conda env create -f .\DATASET_GENERATOR\requirements.yml
```

## 5. Generar un dataset pequeno de prueba

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

Salida esperada:

```text
CUSTOM_5CLASS_SMOKE_TEST/
  TRAIN/
    AUDIO/
    LABELS/
    ANNOTATIONS/
  VAL/
    AUDIO/
    LABELS/
    ANNOTATIONS/
  TEST/
    AUDIO/
    LABELS/
    ANNOTATIONS/
  dataset_config.json
  dataset_event_counts.csv
  dataset_event_counts.json
```

## 6. Generar dataset completo comparable al dataset principal

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

Importante:

- No usar como `--output-root` una carpeta que ya contiene archivos.
- El generador evita sobrescribir datasets existentes.
- Usar un nuevo nombre para cada regeneracion comparable.

## 7. Conversion JSON a anotaciones ADTOF `.txt`

Normalmente el generador ya crea anotaciones si se usa `--write-annotations`.

Si necesitas convertir manualmente:

```powershell
conda run -n adtof_rod python .\DATASET_GENERATOR\json_to_adtof_txt.py `
  --root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST
```

Formato de anotaciones:

```text
timestamp_seconds    midi_pitch    velocity
```

Mapeo vigente:

- KD = 35
- SD = 38
- T12 = 47
- T14 = 45
- T16 = 43

## 8. Validacion recomendada despues de generar

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_wav_dataset_format.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --target-sr 12000 `
  --expected-subtype PCM_16 `
  --peak-limit 0.9
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\check_dataset_temporal_consistency.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --duration-tolerance-ms 5 `
  --min-tail-margin-ms 100
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\analyze_dataset_event_counts.py `
  --counts-csv .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST\dataset_event_counts.csv
```

```powershell
conda run -n adtof_rod python .\SCRIPTS\test_custom_dataset_loader.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --tracks-per-split 2 `
  --training-sequence 64 `
  --batch-size 1 `
  --sample-rate 100 `
  --context 9
```

## 9. Dataset principal actual

Dataset principal local:

```text
DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL
```

Configuracion:

- Total: 12.000 frases
- TRAIN: 9.600
- VAL: 1.200
- TEST: 1.200
- Perfiles: `basic_groove`, `standard_rock`, `fill_heavy`, `fast_dense`, `humanized_mixed`
- Pesos: 0.30, 0.30, 0.15, 0.10, 0.15
- `tom_boost`: 1.3
- `target_tom_share`: 0.30
- Peak limit: 0.9
- WAV mono, 12 kHz, PCM16
- Anotaciones `.txt` activadas
- Seed: 43

Este dataset no se sube a GitHub por tamano.

## 10. Que se versiona y que no

Se versiona:

- codigo fuente
- scripts
- README
- documentacion
- registros livianos
- tests

No se versiona:

- `MUESTRAS/`
- datasets generados
- `AUDIO/`
- `LABELS/`
- `ANNOTATIONS/`
- `.wav`
- `.npy`
- checkpoints `.h5`
- resultados pesados

## 11. Errores frecuentes

### Error: no existe `MUESTRAS`

Solucion:

- Descargar desde Drive.
- Ubicar en `DATASET_GENERATOR/MUESTRAS/`.

### Error: el dataset ya existe

Solucion:

- Usar otro `--output-root`.

### Error: no aparecen anotaciones `.txt`

Solucion:

- Usar `--write-annotations`.
- O correr manualmente `json_to_adtof_txt.py`.

### Error: se usa accidentalmente el esquema antiguo

Solucion:

- Revisar que las clases sean KD, SD, T12, T14, T16.
- No usar TT/HH/CY para datasets nuevos.

### Error: rutas con espacios en PowerShell

Solucion:

- Usar comillas para rutas completas.
- Mantener comandos desde la raiz del repositorio.

## 12. Siguiente paso despues de generar dataset

Despues de validar el dataset, el entrenamiento se realiza desde `SCRIPTS/mini_train_custom_split.py`. Ejemplo corto:

```powershell
conda run -n adtof_rod python .\SCRIPTS\mini_train_custom_split.py `
  --dataset-root .\DATASET_GENERATOR\CUSTOM_5CLASS_SMOKE_TEST `
  --run-name smoke_from_generated_dataset `
  --epochs 1 `
  --steps-per-epoch 2 `
  --validation-steps 1 `
  --batch-size 1 `
  --training-sequence 64 `
  --context 9 `
  --sample-rate 100 `
  --output-dir .\RUNS
```

Para entrenamientos serios, revisar la documentacion de entrenamiento y usar onset-based F1 como metrica principal de evaluacion.

Mas detalle: `DATASET_GENERATOR/DOCS/README_GENERADOR_DATASET.md`.
