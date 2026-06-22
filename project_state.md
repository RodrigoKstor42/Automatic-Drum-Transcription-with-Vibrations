# Estado del proyecto

Fecha de corte: 2026-06-21.

## Contexto

Este repositorio parte de la replicacion local del metodo ADTOF de M. Zehren para transcripcion automatica de bateria. El objetivo general es usar esa base para avanzar hacia transcripcion de bateria a partir de senales vibratorias captadas con triggers piezoelectricos instalados en las membranas de los cuerpos de la bateria.

El proyecto ya no se limita a audio acustico convencional. La linea actual trabaja con muestras vibratorias individuales de cinco clases:

- KD: bombo, MIDI 35
- SD: caja, MIDI 38
- T12: tom de 12 pulgadas, MIDI 47
- T14: tom de 14 pulgadas, MIDI 45
- T16: tom de 16 pulgadas, MIDI 43

Estas clases estan declaradas en `adtof/config.py` como `VIBRO_LABELS_5`, `VIBRO_LABELS_5TXT` y `VIBRO_WEIGHTS_5`. El DataLoader de ADTOF tambien fue adaptado para un escenario `custom_split`, pensado para datasets con carpetas `TRAIN`, `VAL` y `TEST`, cada una con `AUDIO` y `ANNOTATIONS`.

## Objetivo tecnico actual

Construir un pipeline reproducible para:

1. Generar frases procedurales de bateria usando muestras vibratorias de golpes individuales.
2. Exportar cada frase como WAV mono, 12 kHz, PCM16, con control de peak para evitar clipping.
3. Exportar labels JSON ricos en metadata y anotaciones `.txt` compatibles con ADTOF.
4. Validar formato WAV, consistencia temporal, conteo de eventos y carga del dataset.
5. Entrenar el modelo ADTOF con el esquema vibratorio de 5 clases.
6. Evaluar metricas frame-wise y onset-based para identificar limitaciones del dataset/modelo.

## Avance principal

Se implemento un generador procedural en `PHRASE_GENERATOR/SRC/phrase_generator.py`. El generador evoluciono desde una frase basica KD+SD+KD+SD hacia una generacion masiva por perfiles musicales. Durante el proceso se corrigieron problemas iniciales de distorsion, clipping, golpes desordenados, solapamientos excesivos y truncamiento de colas.

Los perfiles actuales estan definidos en `PHRASE_GENERATOR/SRC/generation_profiles.py`:

- `basic_groove`
- `standard_rock`
- `fill_heavy`
- `fast_dense`
- `humanized_mixed`

Cada perfil controla rango de BPM, minimo/maximo de eventos, probabilidad de fills, probabilidad de toms, densidad ritmica, solapamientos naturales, humanizacion temporal y variacion dinamica.

## Estado del pipeline de datos

La generacion actual usa un pipeline unificado. `PHRASE_GENERATOR/generate_tomboost_dataset.py` se mantiene como wrapper de compatibilidad, pero la logica vive en `PHRASE_GENERATOR/SRC/phrase_generator.py`.

El dataset final mas importante hasta este corte es:

`PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`

Caracteristicas:

- 12,000 frases totales.
- 9,600 TRAIN, 1,200 VAL, 1,200 TEST.
- WAV mono, 12 kHz, PCM16.
- Peak limit: 0.9.
- Anotaciones JSON y `.txt`.
- Perfiles combinados con pesos: basic_groove 0.30, standard_rock 0.30, fill_heavy 0.15, fast_dense 0.10, humanized_mixed 0.15.
- Distribucion objetivo: KD 35%, SD 35%, T12 10%, T14 10%, T16 10%.

Conteo observado en `dataset_event_counts.csv`:

- TRAIN: KD 39008, SD 39007, T12 11145, T14 11145, T16 11145.
- VAL: KD 4870, SD 4870, T12 1392, T14 1392, T16 1392.
- TEST: KD 4891, SD 4890, T12 1397, T14 1397, T16 1397.

## Estado del entrenamiento

El script principal de entrenamiento controlado es `SCRIPTS/mini_train_custom_split.py`. Aunque se llama `mini_train`, ya permite corridas de 10 epochs, checkpoints, evaluacion en TEST, metricas por umbral y evaluacion de onsets.

La corrida mas relevante hasta ahora es:

`RUNS/full_5class_low_10ep_eval`

Configuracion:

- Dataset: `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- Modelo: `Frame_RNN`
- Epochs: 10
- Steps per epoch: 200
- Validation steps: 50
- Batch size: 1
- Training sequence: 64
- Context: 9
- Feature sample rate: 100
- Evaluacion onset: activada, tolerancia 50 ms
- Checkpoints: `best_weights.weights.h5` y `final_weights.weights.h5`

Resultado general:

- La perdida bajo y se estabilizo alrededor de `loss ~= 0.064`, `val_loss ~= 0.065`.
- Las metricas onset muestran que el modelo aprende senales utiles, pero sigue siendo muy sensible al umbral.
- T14 sigue siendo la clase mas debil en evaluacion onset.
- Los mejores umbrales no necesariamente son 0.5; los resultados utiles aparecen mas cerca de 0.03 a 0.1 segun clase.

## Archivos importantes

- `PHRASE_GENERATOR/SRC/phrase_generator.py`: generacion procedural y generacion directa de splits.
- `PHRASE_GENERATOR/SRC/generation_profiles.py`: perfiles musicales.
- `PHRASE_GENERATOR/json_to_adtof_txt.py`: conversion JSON a anotaciones ADTOF `.txt`.
- `SCRIPTS/check_wav_dataset_format.py`: diagnostico tecnico WAV.
- `SCRIPTS/check_dataset_temporal_consistency.py`: diagnostico temporal WAV/JSON/TXT.
- `SCRIPTS/analyze_dataset_event_counts.py`: diagnostico de balance de eventos.
- `SCRIPTS/test_custom_dataset_loader.py`: smoke test del DataLoader `custom_split`.
- `SCRIPTS/mini_train_custom_split.py`: entrenamiento controlado y evaluacion.
- `adtof/model/dataLoader.py`: soporte `custom_split`.
- `adtof/config.py`: vocabulario vibratorio de 5 clases.

## Siguientes pasos sugeridos

1. Repetir entrenamiento con mas pasos por epoch y/o mas epochs usando el dataset `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`.
2. Evaluar calibracion de umbrales por clase, especialmente para T12, T14 y T16.
3. Revisar por que T14 tiende a quedarse con bajo recall/F1 en onset.
4. Comparar una corrida con `peak_limit=0.9` contra una version menos conservadora si se confirma que no hay clipping.
5. Documentar resultados de metricas finales en cada carpeta de `RUNS` para poder comparar experimentos sin abrir todos los CSV manualmente.
