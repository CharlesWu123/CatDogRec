# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/9 20:44 
@File : dataset.py 
@Desc : 
'''
import os

from PIL import Image
from torch.utils.data import Dataset


class CatDogsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(CatDogsDataset, self).__init__()
        self.transform = transform
        self.file_path_list, self.file_label_list = self.get_path_label(img_dir)

    def __getitem__(self, item):
        img = Image.open(self.file_path_list[item])
        if self.transform is not None:
            img = self.transform(img)
        label = self.file_label_list[item]
        return img, label

    def __len__(self):
        return len(self.file_path_list)

    def get_path_label(self, img_dir):
        file_path_list = []
        file_label_list = []
        for cur_dir, next_dir, file_name_list in os.walk(img_dir):
            for file_name in file_name_list:
                name, ext = os.path.splitext(file_name)
                if ext.lower() not in ['.jpg', '.png', '.jpeg']: continue
                file_path = os.path.join(cur_dir, file_name)
                file_path_list.append(file_path)
                if file_name.startswith('cat'):
                    file_label_list.append(0)
                else:
                    file_label_list.append(1)
        return file_path_list, file_label_list