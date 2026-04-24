import os
import torch
import numpy as np

txt_path = "/data4/liandong/PROJECTS/BridgeVoc-open/Datascp/LSJ/ljs_audio_text_test_filelist.txt"
save_path = "/data4/liandong/PROJECTS/BridgeVoc-open/Datascp/LSJ/ljs_audio_text_test_filelist_update.txt"

lines = open(txt_path, 'r').readlines()
save_scp = open(save_path, 'w')
for l in lines:
   cur_filename = l.strip()[7:]  # remove DUMMY2/
   save_scp.write(cur_filename + '\n')
print('Write the scp...')
save_scp.close()


