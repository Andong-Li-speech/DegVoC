import os
import shutil
from tqdm import tqdm

scp_path = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/LibriTTS/dev-clean-other.txt"
source_path = "/data4/liandong/datasets/LibriTTS/LibriTTS"
save_path = "/data4/liandong/datasets/LibriTTS/LibriTTS/dev-clean-other-andong"
if not os.path.exists(save_path):
   os.makedirs(save_path)

lines = open(scp_path, 'r').readlines()
for l in tqdm(lines):
   rela_path = l.strip().split('|')[0]
   abs_path = os.path.join(source_path, rela_path) + ".wav"
   filename = os.path.split(abs_path)[-1]
   shutil.copy(abs_path, os.path.join(save_path, filename))
   