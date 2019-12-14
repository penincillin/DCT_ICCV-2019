from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import os.path as osp
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2
import util.parallel_io as pio
import util.ry_utils as ry_utils
from data.data_preprocess import DataProcessor
import data.data_utils as data_utils


class InferDataset(data.Dataset):

    def __init__(self, opt, data_num=None):
        self.opt = opt
        self.annotation_path = opt.infer_dataset_path
        self.isTrain = False
        self.data_root = opt.data_root
        self.data_list = data_utils.load_from_annotation(self.annotation_path)
        self.update_path()

        # pad data list so that the number of data can be divisible by batch size
        # the redundant data will be removed after the whole dataset has been processed
        add_num = opt.batchSize - len(self.data_list)%opt.batchSize
        self.data_list += self.data_list[:add_num]

        # transform list        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.data_processor = DataProcessor(opt)
    
    def update_path(self):
        for data in self.data_list:
            for key in data.keys():
                if key.find("path")>=0:
                    data[key] = osp.join(self.data_root, data[key])


    def preprocess_data(self, img):
        # pad and resize
        img = self.data_processor.padding_and_resize(img)[0]
        return img


    def __getitem__(self, index):
        single_data = self.data_list[index]
        # load raw data
        img_path = single_data['image_path']
        # open image
        img = cv2.imread(img_path)
        img = self.preprocess_data(img)
        # change numpy.array to torch.tensor
        img = self.transform(img).float()
        data = dict(
            img=img,
            index=torch.tensor(index)
        )
        return data

    def getitem(self, index):
        return self.__getitem__(index)

    def __len__(self):
        return len(self.data_list)

    @property
    def name(self):
        return 'InferDataset'