# Automatic-Drum-Transcription-with-Vibrations
A step by step guide on how to recreate and troubleshoot M. Zehren's ADTOF method.

## Resúmen
El presente proyecto tiene como objetivo replicar el modelo de transcrpción automática de batería ADTOF, también se incluyen soluciones a problemas encontrados en el proceso y sus soluciones.
Posteriormente, se plantea adaptarlo para operar con señales vibratorias captadas mediante sensores piezoeléctricos con la finalidad de demostrar la viabilidad de realizar transcripción de batería no a partir de audio, sino desde la vibración de la misma membrana del instrumento.

## Objetivos
1. Replicar el modelo ADTOF en entorno local
2. Documentar el proceso de instalación
3. Resolver incompatibilidades de dependencias
4. Validar funcionamiento del modelo con archivos .wav
5. Preparar la base para adaptación a señales piezoeléctricas

## Proceso de replicación
Para correr el modelo correctamente, se puede realizar de dos maneras:

### Google Colab Notebook
Se puede correr el modelo con el siguiente Notebook de [Google Colab](https://colab.research.google.com/drive/1G_UeWav_AMaaqfJxR4cVgH5EdrvsDQGT?usp=sharing#scrollTo=glwHAUsGpy9s), esto producira un ejemplo de transcripcion de un video.

### Clonar repositorio directamente a la PC
Es posible clonar el repositorio directamente a una computadora para correrlo localmente con Visual Studio Code, de esta manera es posible cargar un archivo .wav propio y realizar su transcripción.
Este repositorio contiene información para la replicación desde su computadora.

## Requisitos:
Para asegurar la reproducibilidad, este proyecto requiere:

- Python 3.10
- Conda (recomendado)
- pip (incluido con python)
- Git

### Instalación de Conda
Se recomienda instalar miniconda (una versión mas ligera de Anaconda)

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
Este proyecto usa **Python 3.10**, conda instalará python automaticamente.




## Instalación

### Clonar repositorio
En una terminal (CMD/Powershell/Bash), ejecuta:

```
git clone <REPO_URL>
cd Automatic-Drum-Transcription-with-Vibrations
```

También se puede descargar el repositorio completo en formato .zip y extraerlo en la ubicación que se desee.

### Crea entorno de trabajo
En una terminal (CMD /Powershell/Bash), ejecuta:

```Python
conda env create -f environment.yml
conda activate adtof_rod
```
Ahora el entorno de trabajo donde ejecutaremos los demas comandos del repo esta activado y se trabajará en esta misma terminal de ahora en adelante.

### Aplica parche requerido para libreria madmom (⚠️IMPORTANTE)
Existen problemas de compatibilidad entre algunas dependencias del repositorio y la libreria madmom (libreria importante para transcripción automatica de música), ejecuta el siguiente comando en la terminal donde activaste el entorno para que el parche se instale:

```Python
python fix_madmom.py
```

### Instala Jupyter (si es necesario)
Con el entorno activado ejecuta en la terminal:

```Python
pip list
```

Este comando mostrara todas las librerias y dependencias instaladas en el entorno que tenemos activado, si no se muestra Jupyter se deberá instalar con el siguiente comando:

```Python
pip install notebook
```
## Ejecutar el proyecto
🔵OPCIÓN A: Mediante Terminal + Navegador
1. Abre una terminal (CMD/Powershell/Bash), abrimos el entorno que instalamos, y ejecuta:

```Python
jupyter notebook
```
Esto abrirá una ventana de Jupyter Notebooks en el navegador predeterminado. Busca la ubicacion donde descargaste el repositorio, una vez en la ubicación del repositorio, navega hasta encontrar:

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

🔵OPCIÓN B: Mediante Microsoft Visual Studio Code
1. Abre el folder que se extrajo del archivo .zip en Visual Studio Code
2. Selecciona el interprete de Python correcto, presiona "*Ctrl + Shift + P*", selecciona "*Python: Select Interpreter*" y elige "*Python 3.10 (adtof_rod)*"
3. Abre el notebook "*drumTranscriptor.ipynb*"
4. Clic en "*Run All*"
- Descarga el repositorio y abre la carpeta correspondiente al repositorio en Visual Studio Code.
- Abre una terminal y ejecuta:


## Ejecución
El repositorio ya viene con 3 archivos .wav de ejemplo para realizar transcripciones de prueba, estos archivos se encuentran en la carpeta audio_files, se producira un grafico de la transcripción resultante y adicionalmente se generará un archivo .mid conteniendo las notas transcritas para ser cargados a un software de interpretación MIDI, estos archivos .mid se guardaran en la carpeta audio_output.

# Generador de frases de batería
El presente proyecto se enfoca en realizar la transcripción automática de baterias mediante el uso de señales vibratorias provenientes de las membranas de los cuerpos de la bateria y capturadas con sensores del tipo piezoelectrico llamados "triggers", y elementos de Machine Learning, de esta manera el sistema podra transcribir golpes con distintas dinamicas. Para lograr esto se grabaron grandes numeros de muestras de golpes individuales de los cuerpos de la bateria, estos cuerpos son el BOMBO (KD), la caja (SD) y los toms de 12, 14 y 16 pulgadas (T12, T14, T16), se grabaron muestras en frecuencias de muestreo de 8KHZ y 12KHZ y con dinamicas de golpes fuertes, medios y suaves o ghost notes.
Este modelo utiliza muestras o frases de bateria, aqui definimos una frase como un fragmento corto de golpes sucesivos de bateria, donde se utilizan distintos elementos de la bateria, en pocas palabras, pueden definirse como pequeñas improvisaciones o fills que usan distintos cuerpos de la bateria, a distintos tempos y dinamicas.
## OBJETIVO
Debido a esta particularidad del modelo, el objetivo especifico de este repo es el de realizar la generacion procedural de estas frases de bateria para el entrenamiento del modelo.

### Crea entorno de trabajo
En una terminal (CMD /Powershell/Bash), ejecuta:

```Python
conda env create -f requirements.yml
conda activate adtof_phrase_generator
```
Ahora el entorno de trabajo donde ejecutaremos los demas comandos del repo esta activado y se trabajará en esta misma terminal de ahora en adelante.


## GENERACION MASIVA POR PERFILES
El generador procedural actual permite seleccionar perfiles reutilizables desde `PHRASE_GENERATOR/SRC/generation_profiles.py`. Cada perfil controla rango de BPM, minimo y maximo de eventos, probabilidad de fills, probabilidad de toms, densidad ritmica, overlaps naturales, humanizacion temporal y variacion dinamica.

🔵Perfiles iniciales disponibles:
- `basic_groove`
- `standard_rock`
- `fill_heavy`
- `fast_dense`
- `humanized_mixed`

Ejemplo para generar 1000 frases usando el perfil `standard_rock`:

```bash
python PHRASE_GENERATOR/SRC/phrase_generator.py --profile standard_rock --count 1000
```

Ejemplo para forzar un BPM fijo y mantener el resto del perfil:

```bash
python PHRASE_GENERATOR/SRC/phrase_generator.py --profile fill_heavy --count 1000 --bpm 120
```

Si no se define `--bpm`, cada frase usa un BPM aleatorio dentro del rango configurado por el perfil. La exportacion WAV y JSON mantiene el pipeline DSP actual: audio mono, 48 kHz, `float32`, normalizacion por peak y labels sincronizados.

