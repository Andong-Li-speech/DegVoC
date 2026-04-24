import os, sys
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import librosa as lib
from tqdm import tqdm
from glob import glob


ref_path = "/data4/liandong/datasets/vocoder/VCTK_out_domain_test_for_vocoder/24k"
pro_path = "/data4/liandong/PROJECTS/FreeV/File_Decodes/VCTK_out_domain/PriorGrad_NFE50"
tar_sr = 24000
meter = pyln.Meter(tar_sr)

file_list = os.listdir(ref_path)
for filename in tqdm(file_list):
   try:
      ref_audio, _ = sf.read(os.path.join(ref_path, filename))
      pro_audio, _ = sf.read(os.path.join(pro_path, filename))
   except:
      continue
   loudness_ref = meter.integrated_loudness(ref_audio)
   loudness_pro = meter.integrated_loudness(pro_audio)
   delta_loudness = loudness_ref - loudness_pro
   gain = np.power(10.0, delta_loudness / 20.0)
   pro_audio *= gain

   sf.write(os.path.join(pro_path, filename), pro_audio, samplerate=tar_sr)
