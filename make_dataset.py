import os
import shutil
from tqdm import tqdm


# 初赛复赛数据集合并
raw_data_1 = '/tcdata/suichang_round1_train_210120'
raw_data_2 = '/tcdata/suichang_round2_train_210316'
data_DIR = './data_sets/raw_data'
if not os.path.exists(data_DIR):os.makedirs(data_DIR)

names =[f for f in os.listdir(raw_data_1)]
for name in tqdm(names):
    shutil.copy(os.path.join(raw_data_1,name),os.path.join(data_DIR,name))

names =[f for f in os.listdir(raw_data_2)]
for name in tqdm(names):
    shutil.copy(os.path.join(raw_data_2,name),os.path.join(data_DIR,'a'+name))

# 数据集划分
train='./data_sets/train'
val='./data_sets/val'

if not os.path.exists(train):os.makedirs(train)
if not os.path.exists(val):os.makedirs(val)

val_ratio=0.15
val_interval=int((1/val_ratio))
train_size=0
val_size=0
names = [f for f in os.listdir(data_DIR) if f[-3:] == 'tif']
print('Data set partitioning!')
for i in tqdm(range(len(names))):
    name=names[i]
    mask_name=name[:-4]+'.png'
    if i%val_interval==0:
        shutil.copy(os.path.join(data_DIR,name),os.path.join(val,name))
        shutil.copy(os.path.join(data_DIR,mask_name), os.path.join(val, mask_name))
        val_size+=1
    else:
        shutil.copy(os.path.join(data_DIR, name), os.path.join(train, name))
        shutil.copy(os.path.join(data_DIR,mask_name), os.path.join(train, mask_name))
        train_size+=1
print("\ntrain size:{},val size:{}".format(train_size,val_size), '\nThe partitioning of the data_sets is complete!')