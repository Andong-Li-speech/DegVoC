# import numpy as np
# import os
# import random
# from tqdm import tqdm
# from glob import glob


# total_file_list = []

# # for LibriTTS set
# libritts_train_scp = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/LibriTTS/train-full.txt"
# libritts_par_path = "/data4/liandong/datasets/LibriTTS/LibriTTS"

# lines = open(libritts_train_scp, 'r').readlines()
# for l in lines:
#    cur_filename = l.strip().split('|')[0]
#    cur_wavfile_path = os.path.join(libritts_par_path, f'{cur_filename}.wav')
#    total_file_list.append(cur_wavfile_path)

# # for EARS dataset 
# # ears_par_path = "/data4/liandong/datasets/ears/48k"
# # ears_file_list = glob(f"{ears_par_path}/*/*.wav")
# # total_file_list += ears_file_list

# # for VCTK dataset
# vctk_par_path = "/data4/liandong/datasets/VCTK-Corpus-0.92/48k_wav_mic1"
# vctk_file_list = glob(f"{vctk_par_path}/*/*.wav")
# total_file_list += vctk_file_list

# # for BAPEN dataset
# bapen_par_path = "/data4/liandong/datasets/BAPEN_24k"
# bapen_file_list = glob(f"{bapen_par_path}/*/*/*.wav")
# total_file_list += bapen_file_list

# # for DNS dataset
# dns_par_path = "/data4/liandong/datasets/DNS-4/datasets_fullband/clean_fullband/datasets_fullband/clean_fullband"
# dns_file_list = glob(f"{dns_par_path}/read_speech/*.wav")
# total_file_list += dns_file_list

# # for MTG dataset
# mtg_par_path = "/data4/liandong/datasets/MTG-Jamendo_24k"
# mtg_file_list = glob(f"{mtg_par_path}/*/*.wav")
# total_file_list += mtg_file_list

# save_par_path = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp"
# save_scp_path = f"{save_par_path}/Libritts_vctk_bapen_dns_mtg_abs_path.scp"
# shuffle_flag = True
# if not shuffle_flag:
#    total_file_list.sort()
# else:
#    random.shuffle(total_file_list)

# train_file_num = len(total_file_list)
# print('The number of files: {}'.format(train_file_num))
# train_scp = open(save_scp_path, 'w')
# for file_idx in tqdm(range(train_file_num)):
#     train_path = total_file_list[file_idx]
#     train_scp.write(train_path+'\n')
# print('Write the scp...')
# train_scp.close()

import numpy as np
import os
import random
from tqdm import tqdm
from glob import glob


total_file_list = []

# for LibriTTS set
libritts_train_scp = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/LibriTTS/train-full.txt"
libritts_par_path = "/data4/liandong/datasets/LibriTTS/LibriTTS"

lines = open(libritts_train_scp, 'r').readlines()
for l in lines:
   cur_filename = l.strip().split('|')[0]
   cur_wavfile_path = os.path.join(libritts_par_path, f'{cur_filename}.wav')
   total_file_list.append(cur_wavfile_path)

# for EARS dataset 
ears_par_path = "/data4/liandong/datasets/ears/24k"
ears_file_list = glob(f"{ears_par_path}/*/*.wav")
if len(ears_file_list) > 0:
   print(f"Load all files of EARS successfully: {len(ears_file_list)}.")
total_file_list += ears_file_list

# for AISHELL3 dataset 
aishell3_par_path = "/data4/liandong/datasets/AISHELL-3/train/wav_24k"
aishell3_file_list = glob(f"{aishell3_par_path}/*/*.wav")
if len(aishell3_file_list) > 0:
   print(f"Load all files of AISHELL3 successfully: {len(aishell3_file_list)}.")
total_file_list += aishell3_file_list

# for VCTK dataset
vctk_par_path = "/data4/liandong/datasets/VCTK-Corpus-0.92/24k_wav_mic1"
vctk_file_list = glob(f"{vctk_par_path}/*/*.wav")
if len(vctk_file_list) > 0:
   print(f"Load all files of VCTK successfully: {len(vctk_file_list)}.")
total_file_list += vctk_file_list

# for BAPEN dataset
bapen_par_path = "/data4/liandong/datasets/BAPEN_24k"
bapen_file_list = glob(f"{bapen_par_path}/*.wav") + \
                  glob(f"{bapen_par_path}/*/*.wav") + \
                  glob(f"{bapen_par_path}/*/*/*.wav") + \
                  glob(f"{bapen_par_path}/*/*/*/*.wav")
if len(bapen_file_list) > 0:
   print(f"Load all files of BAPEN successfully: {len(bapen_file_list)}.")
total_file_list += bapen_file_list

# for DiDiSpeech dataset
didi_par_path = "/data4/liandong/datasets/DiDiSpeech/24k"
didi_file_list = glob(f"{didi_par_path}/*.WAV") + \
                  glob(f"{didi_par_path}/*/*.WAV") + \
                  glob(f"{didi_par_path}/*/*/*.WAV")
if len(didi_file_list) > 0:
   print(f"Load all files of DiDiSpeech successfully: {len(didi_file_list)}.")
total_file_list += didi_file_list

# for DNS dataset
dns_par_path = "/data4/liandong/datasets/DNS-4/datasets_fullband/clean_fullband/datasets_fullband/clean_fullband/read_speech/24k"
dns_file_list = glob(f"{dns_par_path}/*.wav")
if len(dns_file_list) > 0:
   print(f"Load all files of DNS-Challenge successfully: {len(dns_file_list)}.")
total_file_list += dns_file_list

# for MTG dataset
mtg_par_path = "/data4/liandong/datasets/MTG-Jamendo_24k"
mtg_file_list = glob(f"{mtg_par_path}/*/*.wav")
len_ = int(0.1 * len(mtg_file_list))
if len(mtg_file_list) > 0:
   print(f"Load all files of MTG successfully: {len_}.")
total_file_list += mtg_file_list[:len_]


save_par_path = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp"
save_scp_path = f"{save_par_path}/Libritts_ears_vctk_didi_bapen_dns_mtg_24k_sr_abs_path.scp"
shuffle_flag = True
if not shuffle_flag:
   total_file_list.sort()
else:
   random.shuffle(total_file_list)

train_file_num = len(total_file_list)
print('The number of files: {}'.format(train_file_num))
train_scp = open(save_scp_path, 'w')
for file_idx in tqdm(range(train_file_num)):
    train_path = total_file_list[file_idx]
    train_scp.write(train_path+'\n')
print('Write the scp...')
train_scp.close()
