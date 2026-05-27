# CONTEXTO DEL PROYECTO
El presente proyecto se enfoca en realizar la transcripción automática de baterias mediante el uso de señales vibratorias provenientes de las membranas de los cuerpos de la bateria y capturadas con sensores del tipo piezoelectrico llamados "triggers", y elementos de Machine Learning, de esta manera el sistema podra transcribir golpes con distintas dinamicas. Para lograr esto se grabaron grandes numeros de muestras de golpes individuales de los cuerpos de la bateria, estos cuerpos son el BOMBO (KD), la caja (SD) y los toms de 12, 14 y 16 pulgadas (T12, T14, T16), se grabaron muestras en frecuencias de muestreo de 8KHZ y 12KHZ y con dinamicas de golpes fuertes, medios y suaves o ghost notes. El modelo a emplearse para este proyecto es el usado en los papers:
- M. Zehren, M. Alunno, and P. Bientinesi, “ADTOF: A large dataset of non-synthetic music for automatic drum transcription,” in Proceedings of the 22st International Society for Music Information Retrieval Conference, Online, 2021, pp. 818–824.
- Zehren, M.; Alunno, M.; Bientinesi, P. High-Quality and Reproducible Automatic Drum Transcription from Crowdsourced Data. Signals 2023, 4, 768-787. https://doi.org/10.3390/signals4040042

Este modelo utiliza muestras o frases de bateria, aqui definimos una frase como un fragmento corto de golpes sucesivos de bateria, donde se utilizan distintos elementos de la bateria, en pocas palabras, pueden definirse como pequeñas improvisaciones o fills que usan distintos cuerpos de la bateria, a distintos tempos y dinamicas.
# OBJETIVO
Debido a esta particularidad del modelo, el objetivo especifico de este repo es el de realizar la generacion procedural de estas frases de bateria para el entrenamiento del modelo.


# CURRENT OBJECTIVE:
Implementar un generador proceradural de frases de bateria que evite los siguientes problemas:
- Anteriores intentos por realizar el generador de frases resultaron en que las frases creadas se crearan exitosamente pero el resultado tenia demasiada distorsion.
- Estos intentos tambien mostraban que los golpes estaban completamente desordenados y sin sentido musical.
- Tambien se observo que varios de los golpes que se podia distinguir se solapaban uno encima del otro, lo que contribuia a que los audios resultantes y sus respectivas formas de ondas se vieran mas distorsionadas e inentendibles.
Ademas, todos los audios generados deben ser mono, ser exportados en WAV y normalizar una frecuencia de muestreo ideal.
# AVANCES ACTUALES:
-  Ahora se logro crear un generador de frases que genere una frase de prueba.
- Esta frase es un ritmo simple de KD+SD+KD+SD
- La frase creada no contiene distorsion, clipping casi inexistente y la forma de onda es consistente con los golpes que separados del dataset de golpes separados.
- Se consiguió crear las 100 frases, pero ahora es mas notable un truncamiento de los golpes, especialmente al introducir las frases creadas en el software Audacity, se nota que gran parte del decaimiento de la onda es cortado y esto se nota al escucharlo, eso si, la distorsion sigue siendo inexistente y las frases suenan casi naturales.
- Luego de los avances del anterior punto se pudo conseguir que el truncamiento de los golpes sea menor, casi hasta eliminado, ahora el decaimiento de la onda ya no se corta ni se oye cortado.
- Ahora la musicalidad de las frases generadas se siente mas natural y tiene mas coherencia con lo que un baterista tocaria.
- Lo siguiente a realizar tendra que ver con la masificación en la generación de las frases para crear el dataset masivo, pero estas deben estar distribuidas en distintos perfiles que permitan crear un dataset variado que acapare el mayor número de variaciones para el dataset.
- Los perfiles configurados han funcionado bien hasta ahorita, solo que se noto una ligera distorsión con algunos audios, particularmente con los bombos (KD), esto visualmente en Audacity se muestra como un exceso de volumen color rojo que auditivamente equivale a una pequeña distorsion cuando suena un bombo (KD).
- Adicionalmente a la metadata actual, tambien se debe añadir ciertos datos que son importantes.
- Ahora viene una parte muy importante, se tiene que crear un dataset splitter que tome los datos almacenados en la actual carpeta donde se esta guardando tanto los archivos WAV de las frases generadas como sus respectivos archivos JSON correspondientes a sus anotaciones y los distribuya en las carpetas de la main branch llamadas TRAIN, VAL y TEST.
# GENERACION MASIVA POR PERFILES
El generador procedural actual permite seleccionar perfiles reutilizables desde `SRC/generation_profiles.py`. Cada perfil controla rango de BPM, minimo y maximo de eventos, probabilidad de fills, probabilidad de toms, densidad ritmica, overlaps naturales, humanizacion temporal y variacion dinamica.

Perfiles iniciales disponibles:
- `basic_groove`
- `standard_rock`
- `fill_heavy`
- `fast_dense`
- `humanized_mixed`

Ejemplo para generar 1000 frases usando el perfil `standard_rock`:

```bash
python SRC/phrase_generator.py --profile standard_rock --count 1000
```

Ejemplo para forzar un BPM fijo y mantener el resto del perfil:

```bash
python SRC/phrase_generator.py --profile fill_heavy --count 1000 --bpm 120
```

Si no se define `--bpm`, cada frase usa un BPM aleatorio dentro del rango configurado por el perfil. La exportacion WAV y JSON mantiene el pipeline DSP actual: audio mono, 48 kHz, `float32`, normalizacion por peak y labels sincronizados.
