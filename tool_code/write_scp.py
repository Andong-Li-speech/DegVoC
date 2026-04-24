import os
import random
from tqdm import tqdm
import glob


train_dir = "/data4/liandong/PROJECTS/CFVoC/Test_Decode/LibriTTS/latents/WavTokenizer/test_other_all"
train_scp_file  = "/data4/liandong/PROJECTS/CFVoC/DatasetsScp/LibriTTS/test-other-all.scp"

                 
train_file_list = glob.glob(f'{train_dir}/*.wav') + \
                  glob.glob(f'{train_dir}/*/*.wav') + \
                  glob.glob(f'{train_dir}/*/*/*.wav') + \
                  glob.glob(f'{train_dir}/*/*/*/*.wav') + \
                  glob.glob(f'{train_dir}/*.WAV') + \
                  glob.glob(f'{train_dir}/*/*.WAV') + \
                  glob.glob(f'{train_dir}/*/*/*.WAV') + \
                  glob.glob(f'{train_dir}/*/*/*/*.WAV')

shuffle_flag = True
if not shuffle_flag:
    train_file_list.sort()
else:
    random.shuffle(train_file_list)

train_file_num = len(train_file_list)

# train
print('The number of files: {} for training'.format(train_file_num))
train_scp = open(train_scp_file, 'w')
for file_idx in tqdm(range(train_file_num)):
    train_path = train_file_list[file_idx]
    train_scp.write(train_path+'\n')
print('Write the scp...')
train_scp.close()



# train_dir = "/data4/liandong/datasets/BAPEN_48k/48k"
# train_scp_file  = "/data4/liandong/datasets/BAPEN_48k/train_48k_info.scp"
# val_scp_file = "/data4/liandong/datasets/BAPEN_48k/val_48k_info.scp"
# val_num = 500 
                 
# train_file_list = glob.glob(f'{train_dir}/*.wav') + \
#                   glob.glob(f'{train_dir}/*/*.wav') + \
#                   glob.glob(f'{train_dir}/*/*/*.wav') + \
#                   glob.glob(f'{train_dir}/*/*/*/*.wav') + \
#                   glob.glob(f'{train_dir}/*.WAV') + \
#                   glob.glob(f'{train_dir}/*/*.WAV') + \
#                   glob.glob(f'{train_dir}/*/*/*.WAV') + \
#                   glob.glob(f'{train_dir}/*/*/*/*.WAV')

# shuffle_flag = True
# if not shuffle_flag:
#     train_file_list.sort()
# else:
#     random.shuffle(train_file_list)

# train_file_num  = len(train_file_list) - val_num

# # train
# print('The number of files: {} for training'.format(train_file_num))
# train_scp = open(train_scp_file, 'w')
# for file_idx in tqdm(range(train_file_num)):
#     train_path = train_file_list[file_idx]
#     train_scp.write(train_path+'\n')
# print('Write the scp...')
# train_scp.close()

# # validation
# print('The number of files: {} for validation'.format(val_num))
# val_scp = open(val_scp_file, 'w')
# for file_idx in tqdm(range(val_num)):
#     val_path = train_file_list[train_file_num + file_idx]  
#     val_scp.write(val_path+'\n')
# print('Write the scp...')
# val_scp.close()