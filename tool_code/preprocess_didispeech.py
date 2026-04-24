import os
import numpy as np
import random
import soundfile as sf
import librosa as lib
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

source_path = "/data4/liandong/datasets/DiDiSpeech/WAV"
target_sampling_rate = 24000
target_len = 8.0
target_path = "/data4/liandong/datasets/DiDiSpeech_{}k/WAV".format(target_sampling_rate//1000)
if not os.path.exists(target_path):
   os.makedirs(target_path)

source_list = glob(f"{source_path}/*/*/*.WAV")

def process_file(cur_filepath):
   filename = os.path.split(cur_filepath)[-1].split('.')[0]
   par_rela_path = '/'.join(cur_filepath.split('/')[-3:-1])
   x, orig_sr = sf.read(cur_filepath)
   try:
      if orig_sr != target_sampling_rate:
         x = lib.core.resample(x, orig_sr=orig_sr, target_sr=target_sampling_rate)
   except:
      print(f"Invalid resampling, sklip the file: {filename}")
      return

   if len(x) > int(target_len * target_sampling_rate):
      nrep = int(np.ceil(len(x) / int(target_len * target_sampling_rate)))
      start, end = 0, int(target_len * target_sampling_rate)
      cnt = 0
      for _ in range(nrep):
         update_x = x[start: end]
         energy_x = np.abs(update_x).sum()
         if energy_x >= 1e-3:
            update_filename = filename + f'_{cnt+1}'
            update_par_path = target_path + '/' + par_rela_path
            if not os.path.exists(update_par_path):
               os.makedirs(update_par_path)
            update_path = target_path + '/' + par_rela_path + '/' + update_filename + '.wav'
            sf.write(update_path, update_x, samplerate=target_sampling_rate)
            cnt += 1
         start = end
         end = min(start + int(target_len * target_sampling_rate), len(x))
   else:
      energy_x = np.abs(x).sum()
      if energy_x >= 1e-3:
         update_par_path = target_path + '/' + par_rela_path
         if not os.path.exists(update_par_path):
            os.makedirs(update_par_path)
         update_path = target_path + '/' + par_rela_path + '/' + filename + '.wav'
         update_x = x
         sf.write(update_path, update_x, samplerate=target_sampling_rate)


if __name__ == "__main__":
   # num_processes = os.cpu_count()  # 使用 CPU 核心数作为进程数，你也可以手动指定
   num_processes = 16
   with Pool(num_processes) as pool:
      list(tqdm(pool.imap(process_file, source_list), total=len(source_list)))
