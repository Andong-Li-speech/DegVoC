import os
import numpy as np
from tqdm import tqdm
from glob import glob
import random

tr_speaker_list = ['p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p009', 'p010',
                   'p011', 'p012', 'p013', 'p014', 'p015', 'p016', 'p017', 'p018', 'p019', 'p020',
                   'p021', 'p022', 'p023', 'p024', 'p025', 'p026', 'p027', 'p028', 'p029', 'p030',
                   'p031', 'p032', 'p033', 'p034', 'p035', 'p036', 'p037', 'p038', 'p039', 'p040',
                   'p041', 'p042', 'p043', 'p044', 'p045', 'p046', 'p047', 'p048', 'p049', 'p050',
                   'p051', 'p052', 'p053', 'p054', 'p055', 'p056', 'p057', 'p058', 'p059', 'p060',
                   'p061', 'p062', 'p063', 'p064', 'p065', 'p066', 'p067', 'p068', 'p069', 'p070',
                   'p071', 'p072', 'p073', 'p074', 'p075', 'p076', 'p077', 'p078', 'p079', 'p080',
                   'p081', 'p082', 'p083', 'p084', 'p085', 'p086', 'p087', 'p088', 'p089', 'p090',
                   'p091', 'p092', 'p093', 'p094', 'p095', 'p096', 'p097', 'p098', 'p099', 'p100']
val_speaker_list = ['p101', 'p102', 'p103']
test_speaker_list = ['p104', 'p105', 'p106', 'p107']

dataset_path = "/data4/liandong/datasets/ears/48k"
train_scp_file = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/EARS-48k/ears_audio_train_filelist.txt"
val_scp_file = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/EARS-48k/ears_audio_val_filelist.txt"
test_scp_file = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/EARS-48k/ears_audio_test_filelist.txt"
par_path = "/".join(train_scp_file.split('/')[:-1])
if not os.path.exists(par_path):
   os.makedirs(par_path)

speaker_list = os.listdir(dataset_path)
up_tr_speaker_list, up_val_speaker_list, up_test_speaker_list =[], [], []
for tmp in tr_speaker_list:
   if tmp in tr_speaker_list:
      up_tr_speaker_list.append(os.path.join(dataset_path, tmp))
for tmp in val_speaker_list:
   if tmp in val_speaker_list:
      up_val_speaker_list.append(os.path.join(dataset_path, tmp))
for tmp in test_speaker_list:
   if tmp in test_speaker_list:
      up_test_speaker_list.append(os.path.join(dataset_path, tmp))

tr_text_list, val_text_list, test_text_list = [], [], []
for cur_spk_path in up_tr_speaker_list:
   wavs_list = os.listdir(cur_spk_path)
   for rela_wav_filename in wavs_list:
      tr_text_list.append(f"{'/'.join(cur_spk_path.split('/')[6:])}/{rela_wav_filename}"[:-4])

for cur_spk_path in up_val_speaker_list:
   wavs_list = os.listdir(cur_spk_path)
   for rela_wav_filename in wavs_list:
      val_text_list.append(f"{'/'.join(cur_spk_path.split('/')[6:])}/{rela_wav_filename}"[:-4])

for cur_spk_path in up_test_speaker_list:
   wavs_list = os.listdir(cur_spk_path)
   for rela_wav_filename in wavs_list:
      test_text_list.append(f"{'/'.join(cur_spk_path.split('/')[6:])}/{rela_wav_filename}"[:-4])

# save for training set
random.shuffle(tr_text_list)
train_scp = open(train_scp_file, 'w')
for file_idx in tqdm(range(len(tr_text_list))):  
    train_scp.write('DUMMY2/' + tr_text_list[file_idx] + '.wav|'+'\n')
print('Write the scp for training set...')
train_scp.close()

# save for validaiton set
random.shuffle(val_text_list)
val_scp = open(val_scp_file, 'w')
for file_idx in tqdm(range(150)):  
   val_scp.write('DUMMY2/' + val_text_list[file_idx] + '.wav|'+'\n')
print('Write the scp for validation set...')
val_scp.close()

# save for testing set
random.shuffle(test_text_list)
test_scp = open(test_scp_file, 'w')
for file_idx in tqdm(range(500)):  
   test_scp.write('DUMMY2/' + test_text_list[file_idx] + '.wav|'+'\n')
print('Write the scp for testing set...')
test_scp.close()


   

      
