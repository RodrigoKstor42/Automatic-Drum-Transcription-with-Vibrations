# Legacy

Esta carpeta contiene scripts historicos o pruebas locales.

No forman parte del pipeline principal actual.

El pipeline principal actual usa:

- `DATASET_GENERATOR/SRC/phrase_generator.py`
- `DATASET_GENERATOR/SRC/generation_profiles.py`
- `DATASET_GENERATOR/generate_tomboost_dataset.py`
- `DATASET_GENERATOR/json_to_adtof_txt.py`

Contenido historico:

- `dataset_splitter.py`: corresponde al pipeline anterior, donde primero se generaba un dataset raw y luego se dividia en `TRAIN/VAL/TEST`.
- `tester.py`: prueba local no reproducible, con ruta absoluta externa.
- `empty_modules/`: puede contener modulos vacios que quedaron de una modularizacion anterior.

No usar estos scripts para generar datasets nuevos salvo que se este reproduciendo una iteracion historica.
