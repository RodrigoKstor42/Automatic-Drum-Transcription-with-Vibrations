import numpy as np
import soundfile as sf
print(np.max(np.abs(sf.read(r"C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\DATASET_VIBRATIONS\output_dataset\sample_0.wav")[0])))