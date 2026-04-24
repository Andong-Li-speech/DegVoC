import os
import shutil
from tqdm import tqdm

scp_path = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/LSJ/ljs_audio_text_test_filelist.txt"
source_path = "/data4/liandong/datasets/LJSpeech-1.1/wavs"
save_path = "/data4/liandong/datasets/LJSpeech-1.1/test_500"
if not os.path.exists(save_path):
   os.makedirs(save_path)

lines = open(scp_path, 'r').readlines()
for l in tqdm(lines):
   rela_path = l.strip().split('|')[0].split('/')[-1]
   abs_path = os.path.join(source_path, rela_path)
   filename = os.path.split(abs_path)[-1]
   shutil.copy(abs_path, os.path.join(save_path, filename))
   