import os
import shutil
from glob import glob
from tqdm import tqdm

source_dir = "/data4/liandong/datasets/LibriTTS/LibriTTS/test-other"
target_dir = "/data4/liandong/PROJECTS/CFVoC/Test_Decode/LibriTTS/latents/WavTokenizer/test_other_all"

if not os.path.exists(target_dir):
   os.makedirs(target_dir)

file_list = glob(f'{source_dir}/*.wav') + \
             glob(f'{source_dir}/*/*.wav') + \
             glob(f'{source_dir}/*/*/*.wav') + \
             glob(f'{source_dir}/*/*/*/*.wav') + \
             glob(f'{source_dir}/*.WAV') + \
             glob(f'{source_dir}/*/*.WAV')

for cur_file in tqdm(file_list):
   file_name = cur_file.split('/')[-1]
   cur_tar_file_path = os.path.join(target_dir, file_name)
   if not os.path.exists(cur_tar_file_path):
      shutil.copy(cur_file, cur_tar_file_path)
      print(f'Copying {file_name} to {target_dir}')
    