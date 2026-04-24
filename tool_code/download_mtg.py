import os
os.environ["HF_HOME"] = "/data4/liandong/datasets/MTG-Jamendo"  # 将此路径替换为你想要的文件夹路径
from datasets import load_dataset

dataset = load_dataset('rkstgr/mtg-jamendo')