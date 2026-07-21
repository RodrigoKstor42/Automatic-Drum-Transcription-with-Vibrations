# Auditoria de PHRASE_GENERATOR

Fecha de auditoria: 2026-07-09.

Alcance: inspeccion conservadora sin borrar, mover ni modificar codigo funcional o datasets. La revision respeta el esquema vigente `vibro_5_toms`: KD=35, SD=38, T12=47, T14=45, T16=43. La metrica principal del proyecto sigue siendo onset-based F1 con tolerancia temporal; esta auditoria no cambia evaluacion, DataLoader ni entrenamiento.

## 1. Resumen ejecutivo

`PHRASE_GENERATOR` contiene una mezcla de codigo vigente, documentacion, muestras fuente, datasets generados y resultados historicos/intermedios. El generador vigente esta concentrado en `SRC/phrase_generator.py` y sus perfiles en `SRC/generation_profiles.py`. `generate_tomboost_dataset.py` se mantiene como wrapper de compatibilidad que delega en el generador unificado. `json_to_adtof_txt.py` sigue siendo util para convertir labels JSON a anotaciones ADTOF `.txt`.

El dataset principal actual es `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`, con estructura entrenable `TRAIN/VAL/TEST` y subcarpetas `AUDIO/LABELS/ANNOTATIONS`. Los datasets historicos o de debug ocupan bastante espacio local, pero estan ignorados por Git segun `.gitignore`. No se detectaron datasets pesados trackeados por Git dentro de `PHRASE_GENERATOR`; el unico JSON trackeado de datos/resumen observado es `dataset_summary.json`.

La limpieza futura deberia separar codigo, scripts, docs, datasets locales e historial. En esta pasada no se aplico ningun movimiento ni eliminacion.

## 2. Estructura encontrada

Elementos principales en la raiz de `PHRASE_GENERATOR`:

| Ruta | Tipo | Observacion |
|---|---|---|
| `SRC/` | codigo fuente | Generador unificado, perfiles y modulos auxiliares/antiguos. |
| `DOCS/` | documentacion | Documentacion historica del pipeline/dataset/arquitectura. |
| `MUESTRAS/` | datos fuente | Muestras de golpes individuales; ignorado por Git. |
| `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL/` | dataset | Dataset principal actual. |
| `CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG/` | dataset | Dataset debug/intermedio. |
| `CUSTOM_5CLASS_MULTIPROFILE_DEBUG/` | dataset | Dataset debug/intermedio. |
| `CUSTOM_5CLASS_TOMBOOST/` | dataset | Dataset historico de prueba tom-boost. |
| `CUSTOM_NORMALIZED/` | dataset | Dataset historico/intermedio. |
| `GENERATED_RAW/` | dataset raw | Salidas raw antiguas con `AUDIO/LABELS`, sin `ANNOTATIONS`. |
| `TRAIN/`, `VAL/`, `TEST/` | dataset split | Split antiguo en raiz; contiene `AUDIO/LABELS/ANNOTATIONS`. |
| `generate_tomboost_dataset.py` | wrapper | Wrapper vigente de compatibilidad hacia `SRC/phrase_generator.py`. |
| `json_to_adtof_txt.py` | script | Conversor JSON a `.txt` ADTOF. |
| `README.md` | documentacion | Modificado antes de esta auditoria segun Git. |
| `requirements.yml` | entorno | Entorno historico/especifico del generador. |
| `dataset_summary.json` | metadata | Resumen de split antiguo. Trackeado por Git. |
| `tester.py` | prueba local | Script manual con ruta absoluta externa; candidato a archivar/eliminar. |
| `__pycache__/` | generado | Cache Python local; candidato a eliminar despues de confirmacion. |

Archivos `.py` encontrados:

- `PHRASE_GENERATOR/generate_tomboost_dataset.py`
- `PHRASE_GENERATOR/json_to_adtof_txt.py`
- `PHRASE_GENERATOR/tester.py`
- `PHRASE_GENERATOR/SRC/audio_utils.py`
- `PHRASE_GENERATOR/SRC/config.py`
- `PHRASE_GENERATOR/SRC/dataset_splitter.py`
- `PHRASE_GENERATOR/SRC/generation_profiles.py`
- `PHRASE_GENERATOR/SRC/labels.py`
- `PHRASE_GENERATOR/SRC/mixer.py`
- `PHRASE_GENERATOR/SRC/phrase_generator.py`

## 3. Elementos vigentes

| Ruta | Tipo | Estado | Descripcion | Como/cuando se usa |
|---|---|---|---|---|
| `PHRASE_GENERATOR/SRC/phrase_generator.py` | modulo/script | vigente | Generador procedural unificado. Genera splits `TRAIN/VAL/TEST`, `AUDIO/LABELS/ANNOTATIONS`, `dataset_config.json` y conteos. Usa esquema `vibro_5_toms`. | Comando directo o via wrapper para crear datasets entrenables. |
| `PHRASE_GENERATOR/SRC/generation_profiles.py` | modulo | vigente | Define perfiles `basic_groove`, `standard_rock`, `fill_heavy`, `fast_dense`, `humanized_mixed`. | Importado por `SRC/phrase_generator.py`. |
| `PHRASE_GENERATOR/generate_tomboost_dataset.py` | wrapper | vigente | Wrapper de compatibilidad; importa y reexporta funciones desde `SRC/phrase_generator.py`, y ejecuta `main()`. | Usado en `dataset_registry.md`, tests y comandos reproducibles historicos/actuales. |
| `PHRASE_GENERATOR/json_to_adtof_txt.py` | script/modulo | vigente | Convierte labels JSON a anotaciones `.txt` ADTOF. Incluye mapeo vigente KD/SD/T12/T14/T16 y compatibilidad `TT -> 47`. | Usado por `phrase_generator.py` para escribir anotaciones y por `SCRIPTS/test_json_to_adtof_txt.py`. |
| `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` | dataset | vigente | Dataset principal actual, 12,000 frases segun `dataset_registry.md`, con peak limit 0.9 y perfiles multiperfil. | Referenciado por entrenamiento/evaluacion en `training_registry.md`, `project_state.md` y carpetas `RUNS`. |
| `PHRASE_GENERATOR/MUESTRAS` | dataset fuente | vigente/revisar | Muestras de golpes individuales usadas por `SAMPLES_DIR` en el generador. | Entrada por defecto de `SRC/phrase_generator.py`. Mantener local. |
| `PHRASE_GENERATOR/DOCS` | documentacion | vigente/revisar | Documentacion del pipeline anterior y arquitectura. | Consulta humana; revisar si se consolida con docs de raiz. |
| `PHRASE_GENERATOR/README.md` | documentacion | vigente/revisar | Documenta uso del generador y evolucion. | Consulta humana. Tenia cambios previos a esta auditoria. |

## 4. Datasets generados

| Ruta | Estado | Tamano aproximado | Contiene AUDIO/LABELS/ANNOTATIONS | Recomendacion |
|---|---|---:|---|---|
| `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` | principal | 2639.75 MB | si | Mantener local. No eliminar ni mover. Debe seguir ignorado en Git. |
| `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG` | debug | 262.53 MB | si | Archivar o mantener local hasta cerrar comparaciones. No eliminar sin confirmacion humana. |
| `PHRASE_GENERATOR/CUSTOM_5CLASS_MULTIPROFILE_DEBUG` | debug/historico | 615.32 MB | si | Archivar si los resultados ya estan documentados; revisar antes de borrar. |
| `PHRASE_GENERATOR/CUSTOM_5CLASS_TOMBOOST` | historico | 290.73 MB | si | Archivar como iteracion 2; borrar solo si existe respaldo/documentacion suficiente. |
| `PHRASE_GENERATOR/CUSTOM_NORMALIZED` | historico | 1076.97 MB | si | Revisar. Parece iteracion 1 normalizada; posible archivo local. |
| `PHRASE_GENERATOR/GENERATED_RAW` | historico/intermedio | 2130.01 MB | AUDIO/LABELS si, ANNOTATIONS no | Revisar/archivar. Posible salida raw previa al split. |
| `PHRASE_GENERATOR/TRAIN` | historico/intermedio | 1704.73 MB | si | Revisar. Split antiguo en raiz; no tocar hasta confirmar que ningun script lo usa como root por defecto. |
| `PHRASE_GENERATOR/VAL` | historico/intermedio | 213.01 MB | si | Revisar junto con `TRAIN/TEST`. |
| `PHRASE_GENERATOR/TEST` | historico/intermedio | 213.84 MB | si | Revisar junto con `TRAIN/VAL`. |
| `PHRASE_GENERATOR/MUESTRAS` | fuente/revisar | 266.42 MB | no | Mantener local. Es entrada por defecto del generador. |

Metadatos importantes encontrados:

- `dataset_config.json`, `dataset_event_counts.csv` y `dataset_event_counts.json` existen en los datasets `CUSTOM_5CLASS_*`.
- `PHRASE_GENERATOR/dataset_summary.json` existe en la raiz y parece asociado al split antiguo.
- Conteo global por extension dentro de `PHRASE_GENERATOR`: 75,360 `.wav`, 55,009 `.json`, 33,400 `.txt`, 16,000 `.npy`, 10 `.py`, 7 `.pyc`, 4 `.csv`, 4 `.md`, 1 `.yml`.

## 5. Scripts vigentes

| Script | Funcion | Entrada esperada | Salida esperada | Comando tipico de uso |
|---|---|---|---|---|
| `PHRASE_GENERATOR/SRC/phrase_generator.py` | Generar datasets procedurales con esquema `vibro_5_toms`. | `MUESTRAS/` o `--samples-dir`, perfiles, tamanos de split, seed. | Dataset con `TRAIN/VAL/TEST`, `AUDIO/LABELS/ANNOTATIONS`, `dataset_config.json`, `dataset_event_counts.*`. | `conda run -n adtof_rod python .\PHRASE_GENERATOR\SRC\phrase_generator.py --output-root .\PHRASE_GENERATOR\NUEVO_DATASET --num-train N --num-val N --num-test N --profiles basic_groove standard_rock fill_heavy fast_dense humanized_mixed --write-annotations` |
| `PHRASE_GENERATOR/generate_tomboost_dataset.py` | Wrapper de compatibilidad para el generador unificado. | Los mismos argumentos de `phrase_generator.py`. | Los mismos outputs de `phrase_generator.py`. | `conda run -n adtof_rod python .\PHRASE_GENERATOR\generate_tomboost_dataset.py --output-root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL --num-train 9600 --num-val 1200 --num-test 1200 --profiles basic_groove standard_rock fill_heavy fast_dense humanized_mixed --profile-weights 0.3 0.3 0.15 0.1 0.15 --tom-boost 1.3 --target-tom-share 0.30 --balance-individual-toms --peak-limit 0.9 --write-annotations --seed 43` |
| `PHRASE_GENERATOR/json_to_adtof_txt.py` | Convertir JSON labels a anotaciones ADTOF `.txt`. | Dataset root con splits y `LABELS/*.json`. | `ANNOTATIONS/*.txt` por split. | `conda run -n adtof_rod python .\PHRASE_GENERATOR\json_to_adtof_txt.py --root .\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` |
| `PHRASE_GENERATOR/SRC/generation_profiles.py` | Definir perfiles de generacion. | No aplica como CLI. | Objetos `GenerationProfile` importables. | Importado por `phrase_generator.py`. |

## 6. Scripts legacy o candidatos a archivar

| Script | Motivo por el que parece legacy | Riesgo de moverlo | Recomendacion |
|---|---|---|---|
| `PHRASE_GENERATOR/SRC/dataset_splitter.py` | Pertenece al pipeline anterior: toma `GENERATED_RAW/AUDIO` y `GENERATED_RAW/LABELS` y copia a `TRAIN/VAL/TEST`. El generador actual ya genera splits directamente. | Medio. Esta documentado en `dataset_registry.md` y puede ser util para reproducir la iteracion 1. | Archivar como historico, no eliminar todavia. Mantener hasta documentar reemplazo. |
| `PHRASE_GENERATOR/tester.py` | Script manual de una linea con ruta absoluta externa a `DATASET_VIBRATIONS\output_dataset\sample_0.wav`. No parece parte del pipeline. | Bajo. Puede ser una prueba local olvidada. | Candidato a archivar o eliminar despues de confirmacion humana. |
| `PHRASE_GENERATOR/SRC/audio_utils.py` | Archivo vacio. No hay contenido funcional. | Bajo si no hay imports; aun asi revisar historial antes de borrar. | Candidato a eliminar despues de confirmar que no se importa. |
| `PHRASE_GENERATOR/SRC/config.py` | Archivo vacio. Puede ser remanente de una arquitectura previa. | Bajo/medio por nombre generico; revisar imports antes de borrar. | Candidato a archivar/eliminar con confirmacion. |
| `PHRASE_GENERATOR/SRC/labels.py` | Archivo vacio. Puede ser remanente de modularizacion previa. | Bajo/medio por posible import futuro. | Candidato a archivar/eliminar con confirmacion. |
| `PHRASE_GENERATOR/SRC/mixer.py` | Archivo vacio. Puede ser remanente de modularizacion previa. | Bajo/medio por posible import futuro. | Candidato a archivar/eliminar con confirmacion. |

## 7. Candidatos a eliminar

No eliminar nada en esta fase. Los siguientes elementos solo son candidatos para una decision posterior:

| Ruta | Motivo | Evidencia | Riesgo | Accion recomendada |
|---|---|---|---|---|
| `PHRASE_GENERATOR/__pycache__/` | Cache generado por Python. | Carpeta local con `.pyc`; no forma parte del codigo fuente. | Bajo. | Eliminar despues de confirmacion o asegurar ignore. |
| `PHRASE_GENERATOR/tester.py` | Prueba manual con ruta absoluta externa. | Contiene lectura directa de un WAV fuera del repo y `print(max(abs(...)))`. | Bajo. | Eliminar o archivar despues de confirmacion humana. |
| `PHRASE_GENERATOR/SRC/audio_utils.py` | Archivo vacio. | `Get-Content` no mostro contenido. | Bajo/medio. | Eliminar si `rg` confirma que no se importa. |
| `PHRASE_GENERATOR/SRC/config.py` | Archivo vacio. | `Get-Content` no mostro contenido. | Medio por nombre generico. | Revisar imports y archivar/eliminar despues. |
| `PHRASE_GENERATOR/SRC/labels.py` | Archivo vacio. | `Get-Content` no mostro contenido. | Medio por relacion conceptual con labels. | Revisar imports y archivar/eliminar despues. |
| `PHRASE_GENERATOR/SRC/mixer.py` | Archivo vacio. | `Get-Content` no mostro contenido. | Medio por relacion conceptual con mezcla audio. | Revisar imports y archivar/eliminar despues. |
| Datasets historicos/debug | Ocupan varios GB. | `CUSTOM_NORMALIZED`, `GENERATED_RAW`, `CUSTOM_5CLASS_*DEBUG`, `CUSTOM_5CLASS_TOMBOOST`, `TRAIN/VAL/TEST` suman mucho espacio local. | Alto si no hay respaldo o si algun experimento aun los necesita. | No borrar. Primero registrar checksums/respaldo y confirmar dependencias. |

## 8. Riesgos detectados

- Datasets pesados mezclados con codigo dentro de `PHRASE_GENERATOR`. Aunque estan ignorados por Git, aumentan el riesgo de borrado accidental o confusion durante limpieza.
- Multiples generaciones historicas conviven con el dataset principal. Esto dificulta saber cual dataset usar sin consultar `dataset_registry.md`.
- `TRAIN/VAL/TEST` en la raiz de `PHRASE_GENERATOR` pueden confundirse con un dataset principal, pero parecen pertenecer al pipeline anterior.
- `json_to_adtof_txt.py` conserva compatibilidad `TT -> 47`. Esto puede ser util para labels historicos, pero debe quedar documentado como compatibilidad, no como retorno al esquema viejo KD/SD/TT/HH/CY.
- `dataset_splitter.py` tiene una opcion `--clean` que elimina WAV/JSON dentro de splits de salida. No usar sin revisar `output_root`.
- `tester.py` contiene ruta absoluta a un dataset externo; no es reproducible en otros equipos.
- Archivos vacios en `SRC/` pueden hacer creer que existe una arquitectura modular que ya no esta implementada.
- `README.md` de `PHRASE_GENERATOR` estaba modificado antes de esta auditoria; no se debe sobrescribir durante limpieza.
- Hay cambios y archivos no trackeados fuera de `PHRASE_GENERATOR` antes de la auditoria (`RUNS`, scripts, notebook). No mezclar esos cambios con una limpieza del generador.

## 9. Propuesta de estructura limpia

Propuesta futura, sin aplicarla todavia:

```text
PHRASE_GENERATOR/
  SRC/
    phrase_generator.py
    generation_profiles.py
  scripts/
    generate_tomboost_dataset.py
    json_to_adtof_txt.py
  docs/
    AUDIT_PHRASE_GENERATOR.md
    README_PHRASE_GENERATOR.md
    PIPELINE.md
    DATASET.md
    ARCHITECTURE.md
  legacy/
    README.md
    dataset_splitter.py
    tester.py
    empty_modules/
      audio_utils.py
      config.py
      labels.py
      mixer.py
  datasets_local/
    README.md
    MUESTRAS/
    CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL/
    archive/
      CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG/
      CUSTOM_5CLASS_MULTIPROFILE_DEBUG/
      CUSTOM_5CLASS_TOMBOOST/
      CUSTOM_NORMALIZED/
      GENERATED_RAW/
      TRAIN/
      VAL/
      TEST/
```

Notas:

- Si se mueve `generate_tomboost_dataset.py`, mantener un wrapper temporal en la raiz o actualizar tests/documentacion en una fase controlada.
- Si se mueven datasets fuera de `PHRASE_GENERATOR`, actualizar registros, scripts y rutas de entrenamiento en una fase separada.
- `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` debe mantenerse intacto hasta tener respaldo y confirmacion humana.

## 10. Plan de limpieza por fases

### 1. Cambios de dataset

Propuestos para una fase posterior:

- Confirmar que `.gitignore` cubre todos los datasets locales pesados: `MUESTRAS`, `GENERATED_RAW`, `CUSTOM_NORMALIZED`, `CUSTOM_5CLASS_TOMBOOST`, `CUSTOM_5CLASS_MULTIPROFILE_DEBUG`, `CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG`, `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`, `TRAIN`, `VAL`, `TEST`.
- Mantener `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL` como dataset principal.
- Crear un inventario con tamano, conteos y fecha de cada dataset historico antes de archivar o borrar.
- Archivar datasets debug/historicos solo despues de confirmar que los resultados estan documentados en `dataset_registry.md` o `training_registry.md`.
- No tocar `AUDIO`, `LABELS`, `ANNOTATIONS`, `.wav`, `.json`, `.txt`, `.npy`, `.h5` ni `.weights.h5` sin una tarea explicita.

### 2. Cambios de generador

Propuestos para una fase posterior:

- Documentar oficialmente que `SRC/phrase_generator.py` es la implementacion principal y `generate_tomboost_dataset.py` es wrapper de compatibilidad.
- Evaluar mover wrappers a `scripts/`, manteniendo compatibilidad o actualizando tests.
- Marcar `dataset_splitter.py` como legacy en documentacion, porque el generador actual crea splits directamente.
- Revisar y eliminar/archivar archivos vacios de `SRC/` solo despues de confirmar imports.
- Mantener `json_to_adtof_txt.py` por su doble uso: conversor CLI y helper importado por el generador.

### 3. Cambios de DataLoader

No proponer cambios en esta auditoria. No modificar `adtof/model/dataLoader.py`.

### 4. Cambios de entrenamiento

No proponer cambios funcionales en esta auditoria. Solo registrar que `SCRIPTS/mini_train_custom_split.py` y scripts de diagnostico dependen del dataset principal o validan su formato.

### 5. Cambios de evaluacion

No proponer cambios funcionales en esta auditoria. Mantener como criterio principal la evaluacion onset-based F1 con tolerancia temporal.

## Verificaciones realizadas

Comandos de inspeccion ejecutados:

- `git status --short`
- `git status --short -- PHRASE_GENERATOR`
- Listado de raiz de `PHRASE_GENERATOR`
- Busqueda de archivos con `rg --files PHRASE_GENERATOR`
- Busqueda de `.py` dentro de `PHRASE_GENERATOR`
- Resumen de carpetas por cantidad de archivos y tamano aproximado
- Resumen de extensiones dentro de `PHRASE_GENERATOR`
- Busqueda de metadatos `dataset_config.json`, `dataset_event_counts.csv`, `dataset_event_counts.json`, `dataset_summary.json`
- Busqueda de imports/referencias a clases y rutas dentro de `.py`, `.md` y `.json`
- Busqueda de referencias externas en documentos y scripts de raiz
- `git ls-files PHRASE_GENERATOR`
- `git check-ignore -v` para datasets conocidos

Resultado Git observado antes de crear este reporte:

- Dentro de `PHRASE_GENERATOR`, Git mostraba `M PHRASE_GENERATOR/README.md`.
- Los datasets principales/historicos indicados estan ignorados por `.gitignore`.
- No se observaron `.wav`, `.txt`, `.npy`, `.h5` ni `.weights.h5` trackeados dentro de `PHRASE_GENERATOR`.
- `PHRASE_GENERATOR/dataset_summary.json` si esta trackeado.

## Decision recomendada

Conservar:

- `SRC/phrase_generator.py`
- `SRC/generation_profiles.py`
- `generate_tomboost_dataset.py`
- `json_to_adtof_txt.py`
- `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- `MUESTRAS`
- `dataset_config.json`, `dataset_event_counts.csv`, `dataset_event_counts.json`

Revisar:

- `CUSTOM_NORMALIZED`
- `GENERATED_RAW`
- `TRAIN/VAL/TEST`
- `DOCS/`
- `dataset_summary.json`
- archivos vacios de `SRC/`

Archivar potencialmente:

- `CUSTOM_5CLASS_MULTIPROFILE_DEBUG`
- `CUSTOM_5CLASS_MULTIPROFILE_LOW_DENSITY_DEBUG`
- `CUSTOM_5CLASS_TOMBOOST`
- `SRC/dataset_splitter.py`
- `tester.py`

Eliminar solo despues de confirmacion humana:

- `__pycache__/`
- `tester.py`
- archivos vacios de `SRC/` si no hay imports
- datasets historicos/debug solo con respaldo o decision explicita

## Limpieza Fase 1 aplicada

Fecha: 2026-07-09

Cambios aplicados:

- Se creo `PHRASE_GENERATOR/legacy/`.
- Se creo `PHRASE_GENERATOR/legacy/README.md`.
- Se creo `PHRASE_GENERATOR/legacy/empty_modules/`.
- Se movio `SRC/dataset_splitter.py` a `legacy/dataset_splitter.py`.
- Se movio `tester.py` a `legacy/tester.py`.
- Se confirmo que `SRC/audio_utils.py`, `SRC/config.py`, `SRC/labels.py` y `SRC/mixer.py` estaban vacios y sin imports funcionales.
- Se movieron los modulos vacios de `SRC/` a `legacy/empty_modules/`.
- Se eliminaron caches `__pycache__` autorizados: `PHRASE_GENERATOR/__pycache__/`, `PHRASE_GENERATOR/SRC/__pycache__/` y `PHRASE_GENERATOR/legacy/__pycache__/`, si existian.

Cambios NO aplicados:

- No se movieron datasets.
- No se borraron datasets.
- No se modifico el generador vigente.
- No se modifico el DataLoader.
- No se modificaron scripts de entrenamiento.
- No se modifico la evaluacion.

## Limpieza Fase 1 aplicada - datasets de prueba

Fecha: 2026-07-09

Nota de ruta: al momento de esta limpieza, el contenido antes auditado como `PHRASE_GENERATOR` se encuentra en `DATASET_GENERATOR`. La limpieza se aplico sobre `DATASET_GENERATOR` respetando las mismas reglas conservadoras.

Cambios aplicados:

- Se creo `DATASET_GENERATOR/DATASETS DE PRUEBA/`.
- Se agregaron reglas `.gitignore` para `PHRASE_GENERATOR/DATASETS DE PRUEBA/`, `PHRASE_GENERATOR/DATASETS_DE_PRUEBA/`, `DATASET_GENERATOR/DATASETS DE PRUEBA/` y `DATASET_GENERATOR/DATASETS_DE_PRUEBA/`.
- Se movieron datasets historicos/debug a `DATASETS DE PRUEBA/`.
- Se movio `GENERATED_RAW` a `DATASETS DE PRUEBA/GENERATED_RAW`.
- Se creo `DATASETS DE PRUEBA/SPLIT_ANTIGUO_ROOT/`.
- Se movio `dataset_summary.json` a `DATASETS DE PRUEBA/SPLIT_ANTIGUO_ROOT/`.
- No se encontraron `TRAIN/VAL/TEST` en la raiz de `DATASET_GENERATOR` al momento de esta fase; si ya habian sido movidos previamente, quedan fuera de la raiz.
- Se creo `DATASETS DE PRUEBA/README.md`.
- Se verifico que `legacy/` y `legacy/empty_modules/` ya existian.
- Se verifico que `dataset_splitter.py`, `tester.py` y los modulos vacios ya estaban en `legacy/`.
- Se eliminaron caches `__pycache__` dentro de `DATASET_GENERATOR`.

Cambios NO aplicados:

- No se toco `CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`.
- No se toco `MUESTRAS`.
- No se modifico el generador vigente.
- No se modifico el DataLoader.
- No se modificaron scripts de entrenamiento.
- No se modifico la evaluacion.
- No se hizo commit.
- No se hizo push.
