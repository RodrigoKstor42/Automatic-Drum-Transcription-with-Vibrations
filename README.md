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

# Proceso de replicación
Para correr el modelo correctamente, se puede realizar de dos maneras:

## Google Colab Notebook
Se puede correr el modelo con el siguiente Notebook de [Google Colab](https://colab.research.google.com/drive/1G_UeWav_AMaaqfJxR4cVgH5EdrvsDQGT?usp=sharing#scrollTo=glwHAUsGpy9s), esto producira un ejemplo de transcripcion de un video.

## Clonar repositorio directamente a la PC
Es posible clonar el repositorio directamente a una computadora para correrlo localmente con Visual Studio Code, de esta manera es posible cargar un archivo .wav propio y realizar su transcripción.
Este repositorio fue probado con python 3.10.
### Requisitos:
Para asegurar la reproducibilidad, este proyecto requiere:

- Python 3.10
- Conda (recomendado)
- pip (incluido con python)
- Git

# Instalación de Conda
Se recomienda instalar miniconda (una versión mas ligera de Anaconda)

1. Windows/macOS/linux
- Ve al siguiente enlace: https://docs.conda.io/en/latest/miniconda.html
- Descarga el instalador para tu sistema operativo
- Instala con la configuración predeterminada

2. Verifica la instalación:
Abre una terminal (CMD/Powershell/Bash) y ejecuta:

```python
conda --version
```
# Instalación de Git
Descarga Git desde: https://git-scm.com/

# Versión de Python
Este proyecto usa Python 3.10, conda instalará python automaticamente.




### Instalación:
Este proyecto puede ser instalado y ejecutado desde Microsoft Visual Studio Code o desde una terminal cmd en su computadora.

## Microsoft Visual Studio Code
- Descarga el repositorio y abre la carpeta correspondiente al repositorio en Visual Studio Code.
- Abre una terminal y ejecuta:

```python
conda env create -f environment.yml
```

- El archivo environment.yml contiene todas dependencias especificas para que el repositorio corra correctamente.
  
- Activa el entorno ejecutando la siguiente linea en la misma terminal: 

```python
conda activate adtof_rod
```

- ⚠️ IMPORTANTE: este proyecto utiliza la libreria madmom la cual presenta conflictos con otras librerias por lo que se deberá ejecutar el siguiente parche, de igual manera en la misma terminal y con el entorno activado, ejecuta:

```python
python fix_madmom.py
```

### Ejecución:
El repositorio ya viene con 3 archivos .wav de ejemplo para realizar transcripciones de prueba, estos archivos se encuentran en la carpeta audio_files, se producira un grafico de la transcripción resultante y adicionalmente se generará un archivo .mid conteniendo las notas transcritas para ser cargados a un software de interpretación MIDI, estos archivos .mid se guardaran en la carpeta audio_output.
