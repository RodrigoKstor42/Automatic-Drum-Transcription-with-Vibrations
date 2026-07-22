# Automatic-Drum-Transcription-with-Vibrations

Repositorio de tesis para reproducir y adaptar ADTOF a transcripcion automatica de bateria usando senales vibratorias/piezos.

## Estado actual del proyecto

El pipeline vigente trabaja con un dataset propio generado proceduralmente desde golpes reales y con el esquema `vibro_5_toms`:

- KD = 35
- SD = 38
- T12 = 47
- T14 = 45
- T16 = 43

La metrica principal para comparar modelos es onset-based F1 con tolerancia temporal. Para reproducir el flujo actual en Windows, VS Code, PowerShell y Conda, usar como referencia principal:

- `DATASET_GENERATOR/README.md`
- `SCRIPTS/COMMANDS_UPDATED.md`
- `RUNS/current_official_baseline.md`
- `RUNS/RUNS_OUTPUT_AUDIT.md`

Las secciones siguientes conservan contexto historico de la replicacion inicial de ADTOF y de las primeras pruebas del generador. Cuando haya conflicto entre rutas antiguas y rutas actuales, priorizar `DATASET_GENERATOR` y los comandos de `SCRIPTS/COMMANDS_UPDATED.md`.

## Resumen
El proyecto comenzo como una replicacion local del modelo ADTOF de M. Zehren para transcripcion automatica de bateria. A partir de esa base, se adapto el flujo para trabajar con senales vibratorias captadas por sensores piezoelectricos, con el fin de evaluar si es posible transcribir golpes de bateria desde la vibracion de la membrana y no solo desde audio acustico.

## Objetivos
1. Replicar el modelo ADTOF en entorno local
2. Documentar el proceso de instalación
3. Resolver incompatibilidades de dependencias
4. Validar funcionamiento del modelo con archivos .wav
5. Preparar la base para adaptación a señales piezoeléctricas

## Proceso de replicacion inicial
Para correr la version base del modelo, hay dos caminos:

### Google Colab Notebook
El modelo base puede ejecutarse con este Notebook de [Google Colab](https://colab.research.google.com/drive/1G_UeWav_AMaaqfJxR4cVgH5EdrvsDQGT?usp=sharing#scrollTo=glwHAUsGpy9s). Produce un ejemplo de transcripcion a partir de un video.

### Clonar repositorio directamente a la PC
Tambien se puede clonar el repositorio y ejecutarlo localmente con VS Code. Esta opcion permite cargar archivos `.wav` propios y revisar la transcripcion generada.

## Requisitos:
Para asegurar la reproducibilidad, este proyecto requiere:

- Python 3.10
- Conda (recomendado)
- pip (incluido con python)
- Git

### Instalación de Conda
Se recomienda instalar Miniconda, que es una version ligera de Anaconda.

1. Windows/macOS/linux
- Ve al siguiente enlace: https://docs.conda.io/en/latest/miniconda.html
- Descarga el instalador para tu sistema operativo
- Instala con la configuración predeterminada

2. Verifica la instalación:
Abre una terminal (CMD/Powershell/Bash) y ejecuta:

```
conda --version
```
### Instalación de Git
Descarga Git desde: https://git-scm.com/

Verifica la instalación desde una terminal (CMD/Powershell/Bash) con:

```
git --version
```

### Versión de Python
Este proyecto usa **Python 3.10**; Conda instalara Python automaticamente.




## Instalación

### Clonar repositorio
En una terminal (CMD/Powershell/Bash), ejecuta:

```
git clone <REPO_URL>
cd Automatic-Drum-Transcription-with-Vibrations
```

También se puede descargar el repositorio completo en formato .zip y extraerlo en la ubicación que se desee.

### Crear entorno de trabajo
En una terminal (CMD /Powershell/Bash), ejecuta:

```Python
conda env create -f environment.yml
conda activate adtof_rod
```
Despues de activar el entorno, ejecutar los comandos restantes desde la misma terminal.

### Aplicar parche requerido para la libreria madmom
Algunas dependencias del repositorio presentan incompatibilidades con `madmom`, una libreria usada en transcripcion automatica de musica. Ejecuta el siguiente comando en la terminal donde activaste el entorno:

```Python
python fix_madmom.py
```

### Instala Jupyter (si es necesario)
Con el entorno activado ejecuta en la terminal:

```Python
pip list
```

Este comando muestra las librerias y dependencias instaladas en el entorno activo. Si Jupyter no aparece en la lista, instalarlo con:

```Python
pip install notebook
```
## Ejecutar el proyecto base

### Opcion A: terminal y navegador
1. Abre una terminal (CMD/Powershell/Bash), abrimos el entorno que instalamos, y ejecuta:

```Python
jupyter notebook
```
Esto abrira una ventana de Jupyter Notebook en el navegador predeterminado. Busca la carpeta donde descargaste el repositorio y navega hasta encontrar:

```
bin/drumTranscriptor.ipynb
```

2. Selecciona el kernel correcto
En Jupyter:
- Haz clic en "*Kernel*" y selecciona "*Change Kernel*"
- Selecciona:
```
Python (adtof_rod)
```

3. Corre las celdas
Se puede correr todas las celdas a la vez seleccionando la opcion "*Run*" y haciendo clic en "*Run All*"

### Opcion B: VS Code
1. Abre el folder que se extrajo del archivo .zip en Visual Studio Code
2. Selecciona el interprete de Python correcto, presiona "*Ctrl + Shift + P*", selecciona "*Python: Select Interpreter*" y elige "*Python 3.10 (adtof_rod)*"
3. Abre el notebook "*drumTranscriptor.ipynb*"
4. Clic en "*Run All*"
- Descarga el repositorio y abre la carpeta correspondiente al repositorio en Visual Studio Code.
- Abre una terminal y ejecuta:


## Ejecución
El repositorio incluye 3 archivos `.wav` de ejemplo para pruebas de transcripcion. Estos archivos estan en `audio_files`; al ejecutarlos se genera un grafico de la transcripcion y un archivo `.mid` con las notas transcritas. Los `.mid` se guardan en `audio_output`.

# Generador de frases de bateria
El proyecto usa senales vibratorias captadas desde las membranas de la bateria mediante sensores piezoelectricos. Para entrenar el modelo se grabaron golpes individuales de KD, SD, T12, T14 y T16 con distintas dinamicas, y luego se combinaron proceduralmente en frases cortas.

En este contexto, una frase es una secuencia breve de golpes de bateria, parecida a un groove, una variacion o un fill. El objetivo del generador es producir muchas frases con tempos, dinamicas y combinaciones distintas, manteniendo labels sincronizados para entrenamiento.

## Objetivo del generador
Generar datasets reproducibles para entrenar ADTOF con el esquema vigente `vibro_5_toms`.

### Crea entorno de trabajo
En una terminal (CMD /Powershell/Bash), ejecuta:

```Python
conda env create -f requirements.yml
conda activate adtof_phrase_generator
```
Despues de activar el entorno, continuar en la misma terminal para los demas comandos.
 ### DESCARGA DE DATASET GENERADO
 Debido a que el dataset donde se encuentran las muestras de datos vibracionales de cada elemento de la bateria es muy extenso, se debe descargar desde: [DATASET GENERADO](https://drive.google.com/drive/folders/1Foz00qrkVIi96EjonxatYKqrwOyJMr42?usp=sharing)

 Una vez descargada la carpeta correspondiente al dataset generado, el contenido debe extraerse dentro de `PHRASE_GENERATOR`. Esta instruccion pertenece al flujo historico; para el flujo vigente usar `DATASET_GENERATOR`.

## GENERACION MASIVA POR PERFILES
El generador procedural actual permite seleccionar perfiles reutilizables desde `PHRASE_GENERATOR/SRC/generation_profiles.py`. Cada perfil controla rango de BPM, minimo y maximo de eventos, probabilidad de fills, probabilidad de toms, densidad ritmica, overlaps naturales, humanizacion temporal y variacion dinamica.

🔵Perfiles iniciales disponibles:
- `basic_groove`
- `standard_rock`
- `fill_heavy`
- `fast_dense`
- `humanized_mixed`

Ejemplo para generar 1000 frases usando el perfil `standard_rock`:

```powershell
conda run -n drum_tools python .\PHRASE_GENERATOR\SRC\phrase_generator.py `
  --profile standard_rock `
  --count 1000 `
  --output-root .\PHRASE_GENERATOR\CUSTOM_NORMALIZED `
  --target-sr 12000 `
  --wav-subtype PCM_16 `
  --peak-limit 0.98
```

Ejemplo para forzar un BPM fijo y mantener el resto del perfil:

```bash
python PHRASE_GENERATOR/SRC/phrase_generator.py --profile fill_heavy --count 1000 --bpm 120
```

Si no se define `--bpm`, cada frase usa un BPM aleatorio dentro del rango configurado por el perfil. La exportacion crea `AUDIO` y `LABELS` dentro de `--output-root`. Los WAV se escriben en mono, 12 kHz y PCM16; solo se reduce la ganancia cuando el peak supera `--peak-limit`. La cantidad de frames no cambia, por lo que los JSON permanecen sincronizados.

Para crear los splits finales:

```powershell
conda run -n drum_tools python .\PHRASE_GENERATOR\SRC\dataset_splitter.py `
  --raw-audio-dir .\CUSTOM_NORMALIZED\AUDIO `
  --raw-labels-dir .\CUSTOM_NORMALIZED\LABELS `
  --output-root .\CUSTOM_NORMALIZED_SPLIT `
  --summary-path .\CUSTOM_NORMALIZED_SPLIT\dataset_summary.json `
  --clean

conda run -n drum_tools python .\PHRASE_GENERATOR\json_to_adtof_txt.py `
  --root .\CUSTOM_NORMALIZED_SPLIT
```

Verificacion tecnica de los WAV:

```powershell
conda run -n drum_tools python .\scripts\check_wav_dataset_format.py `
  --dataset-root .\CUSTOM_NORMALIZED_SPLIT
```
