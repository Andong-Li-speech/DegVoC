import os
import numpy as np
import shutil
import math
import soundfile as sf
from tqdm import tqdm
from glob import glob
import torchaudio

wav_path = "/data4/liandong/datasets/VCTK-Corpus-0.92/48k_wav_mic1"
wav_path_list = glob(f"{wav_path}/*/*.wav")
for cur_wav_path in tqdm(wav_path_list):
   x, sr = sf.read(cur_wav_path)
   if np.isnan(np.mean(x)) or np.isinf(np.mean(x)) or np.mean(np.abs(x)) < 1e-10:
      os.remove(cur_wav_path)
      cur_wav_filename = os.path.split(cur_wav_path)[-1]
      print(f"The file {cur_wav_filename} has been deleted.")
   else:
      pass
   # try:
   #    x, sr = sf.read(cur_wav_path)
   #    # x, sr = torchaudio(cur_wav_path)
      
   # except:
   #    os.remove(cur_wav_path)
   #    cur_wav_filename = os.path.split(cur_wav_path)[-1]
   #    print(f"The file {cur_wav_filename} has been deleted.")   
