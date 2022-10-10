# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/10 12:26
@File: data_process.py
@Desc: 
"""
import os
import random
import shutil
data_dir = './train'
dst_test_dir = './test_set'
dst_train_dir = './data/train_set'
os.makedirs(dst_test_dir, exist_ok=True)
os.makedirs(dst_train_dir, exist_ok=True)
file_list = os.listdir(data_dir)
print(len(file_list))
label_data = {'cat': [], 'dog': []}
for file_name in file_list:
    if file_name.startswith('cat'):
        label_data['cat'].append(file_name)
    else:
        label_data['dog'].append(file_name)
print(len(label_data['cat']), len(label_data['dog']))
test_data = {}
test_data['cat'] = random.sample(label_data['cat'], 2500)
test_data['dog'] = random.sample(label_data['dog'], 2500)
for label in ['cat', 'dog']:
    for file_name in test_data[label]:
        file_path = os.path.join(data_dir, file_name)
        dst_file_path = os.path.join(dst_test_dir, file_name)
        shutil.move(file_path, dst_file_path)
for file_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file_name)
    dst_file_path = os.path.join(dst_train_dir, file_name)
    shutil.move(file_path, dst_file_path)