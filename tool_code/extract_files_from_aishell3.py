import os
import shutil
import random
import soundfile as sf
import librosa as lib
from glob import glob
from tqdm import tqdm

source_dir = "/data4/liandong/datasets/AISHELL-3/test/wav"
dec_dir = "/data4/liandong/datasets/vocoder/aishell3_out_domain_test_fopr_vocoder"
if not os.path.exists(dec_dir):
   os.makedirs(dec_dir)

num_files = 200
target_sampling_rate = 24000

file_list = glob(f"{source_dir}/*.wav") + \
            glob(f"{source_dir}/*/*.wav") + \
            glob(f"{source_dir}/*/*/*.wav")
random.shuffle(file_list)
if num_files:
   file_list_ = file_list[:num_files]
else:
   file_list_ = file_list

for cur_file_path in tqdm(file_list_):
   filename = os.path.split(cur_file_path)[-1]
   inpt_x, orig_sr = sf.read(cur_file_path)
   if orig_sr != target_sampling_rate:
      inpt_x = lib.core.resample(inpt_x, orig_sr=orig_sr, target_sr=target_sampling_rate)
   
   sf.write(os.path.join(dec_dir, filename), inpt_x, samplerate=target_sampling_rate)

print("Copy has finished!")
