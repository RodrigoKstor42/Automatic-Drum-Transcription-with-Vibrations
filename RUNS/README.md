# RUNS

Esta carpeta contiene outputs experimentales del proyecto de tesis ADTOF adaptado a senales vibratorias/piezos.

Contexto vigente:

- Esquema: `vibro_5_toms`
- MIDI: KD=35, SD=38, T12=47, T14=45, T16=43
- Metrica principal: onset-based F1 con tolerancia temporal
- Dataset principal actual: `DATASET_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`

## Runs visibles en raiz

Estas carpetas se mantienen en la raiz porque son baseline actual, candidata fuerte, baseline previa o candidatas/comparaciones recientes que aun estan referenciadas por artefactos de comparacion:

- `serious_5class_low_60ep_weighted_t14t16`: baseline oficial actual. Ver `current_official_baseline.md` y `current_official_baseline.json`.
- `serious_5class_low_60ep_focal_soft_toms_v2`: candidata fuerte global; mejora micro-F1 global, con trade-off en toms.
- `serious_5class_low_60ep_calibrated`: baseline previa para trazabilidad.
- `serious_5class_low_60ep_focal_soft_toms_v3`: candidata reciente con comparacion propia.
- `serious_5class_low_60ep_focal_toms_v1`: candidata secundaria comparada contra baseline oficial.
- `serious_5class_low_60ep_weighted_sd_toms_v1`: candidata secundaria comparada contra baseline oficial.
- `serious_5class_low_60ep_weighted_sd_light_keep_toms_v2`: candidata/referencia a revisar, citada por comparaciones.
- `serious_5class_low_60ep_weighted_balanced_v2`: output parcial/calibracion a revisar, citado por comparaciones.
- `serious_5class_low_60ep_weighted_balanced_v3`: candidata a revisar, citada por comparaciones.
- `COMPARISONS`: comparaciones entre baseline y candidatas.

## Documentos clave

- `RUNS_OUTPUT_AUDIT.md`: auditoria enfocada de outputs y clasificacion experimental.
- `current_official_baseline.md`: resumen humano de la baseline oficial actual.
- `current_official_baseline.json`: registro estructurado de la baseline oficial actual.

## Legacy

`legacy/` conserva runs smoke, debug, historicas o no clasificadas con seguridad. No se borro ningun resultado.

- `legacy/smoke_debug/`: pruebas smoke/debug, ejecuciones de 1 epoca y salidas temporales.
- `legacy/historical/`: baselines/evaluaciones antiguas que no son baseline oficial actual.
- `legacy/review/`: outputs no clasificados con seguridad.

## Nota sobre referencias

Antes de mover carpetas se buscaron referencias livianas en documentacion/configs. Algunas referencias historicas siguen apuntando a rutas antiguas en raiz, especialmente en `training_registry.md` y `project_state.md`. Las comparaciones actuales mantienen rutas absolutas hacia las runs serias, por eso esas runs permanecen en la raiz.
