# Automatic-Drum-Transcription-with-Vibrations
A step by step guide on how to recreate and troubleshoot M. Zehren's ADTOF method.
## Resúmen
El presente proyecto tiene como objetivo replicar el modelo de transcrpción automática de batería ADTOF, también se incluyen soluciones a problemas encontrados en el proceso y sus soluciones.
Posteriormente, se plantea adaptarlo para operar con señales vibratorias captadas mediante sensores piezoeléctricos con la finalidad de demostrar la viabilidad de realizar transcripción de batería no a partir de audio, sino desde la vibración de la misma membrana del instrumento.
## Objetivos
- Replicar el modelo ADTOF en entorno local
- Documentar el proceso de instalación
- Resolver incompatibilidades de dependencias
- Validar funcionamiento del modelo con archivos .wav
- Preparar la base para adaptación a señales piezoeléctricas
# Proceso de replicación
Para correr el modelo correctamente, se puede realizar de dos maneras:
## Google Colab Notebook
Se puede correr el modelo con el siguiente Notebook de Google Colab, esto producira un ejemplo de transcripcion de un video.
https://colab.research.google.com/drive/1G_UeWav_AMaaqfJxR4cVgH5EdrvsDQGT?usp=sharing#scrollTo=glwHAUsGpy9s
## Clonar repositorio directamente a la PC
Es posible clonar el repositorio directamente a una computadora para correrlo localmente con Visual Studio Code, de esta manera es posible cargar un archivo .wav propio y realizar su transcripción.
Este repositorio fue probado con python 3.10.
### Requisitos:
- Conda
- Python 3.10
### Instalación:
- Descarga el repositorio y abrelo en Visual Studio Code.
- Abre una terminal y ejecuta: conda env create -f environment.yml
- El archivo environment.yml contiene todas dependencias especificas para que el repositorio corra correctamente.
- Activa el con: conda activate adtof_env
### Ejecución:
El repositorio ya viene con 3 archivos .wav de ejemplo para realizar transcripciones de prueba, estos archivos se encuentran en la carpeta audio_files, se producira un grafico de la transcripción resultante y adicionalmente se generará un archivo .mid conteniendo las notas transcritas para ser cargados a un software de interpretación MIDI, estos archivos .mid se guardaran en la carpeta audio_output.
