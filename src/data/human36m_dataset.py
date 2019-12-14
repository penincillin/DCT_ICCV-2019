from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from PIL import Image
from scipy import misc
import cv2
import pickle
from PIL import Image, ImageDraw
from data.data_preprocess import DataProcessor
import copy

class Human36MDataset(BaseDataset):

    def __init__(self, opt, data_num=0):
        self.opt = opt
        self.annotation_path = opt.human36m_anno_path
        self.isTrain = opt.isTrain

        data_list = self.load_annotation(self.annotation_path)
        data_list = sorted(data_list, key=lambda a:a['image_path'])
        self.update_path(opt.data_root, data_list)

        if opt.isTrain and data_num > 0:
            data_num = min(data_num, len(data_list))
            data_list = random.sample(data_list, data_num)
        self.data_list = data_list

        transform_list = [ transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.data_processor = DataProcessor(opt)
    

    def preprocess_data(self, img, IUV, keypoints, smpl_pose):
        # pad and resize, 
        #two '_' stands for dp_kps and dp_kps_weight
        img, kps, kps_weight, IUV, _, _ = \
            self.data_processor.padding_and_resize(img, keypoints, IUV)

        # random flip, only do in training phase
        # two '_' stands for densepose annotatoins that human3.6m dataset does not contain
        if self.isTrain:
            img, kps, kps_weight, IUV, _, _, _, smpl_pose, flipped = \
                self.data_processor.random_flip(img, kps, kps_weight, IUV, \
                                                None, None, None, smpl_pose)
        # normalize coords of keypoinst to [-1, 1]
        # '_' stands for densepose keypoints that human3.6m dataset does not contain
        kps, _ = self.data_processor.normalize_keypoints(kps, None)
        # return the results
        return img, IUV, kps, kps_weight, smpl_pose


    def __getitem__(self, index):
        # load raw data
        single_data = self.data_list[index]
        # image
        img_path = single_data['image_path']
        iuv_path = single_data['IUV_path']
        # other data
        keypoints = single_data['joints_2d']
        smpl_pose = single_data['smpl_pose']
        smpl_shape = single_data['smpl_shape']

        # open images and IUV
        img = cv2.imread(img_path)
        IUV = cv2.imread(iuv_path)
        keypoints = keypoints.T

        # preprocess the images and the corresponding annotation
        img, IUV, kps, kps_weight, smpl_pose = \
            self.preprocess_data(img, IUV, keypoints, smpl_pose)

        img = self.transform(img).float()
        IUV = self.data_processor.transform_IUV(IUV).float()
        kps = torch.from_numpy(kps).float()
        kps_weight = torch.from_numpy(kps_weight).float()
        smpl_pose = torch.from_numpy(smpl_pose).float()
        smpl_shape = torch.from_numpy(smpl_shape).float()
        smpl_params_weight = torch.ones((1,), dtype=torch.float32)
        
        result = dict(
            img = img,
            IUV = IUV,
            keypoints = kps,
            keypoints_weights = kps_weight,
            smpl_shape = smpl_shape,
            smpl_pose = smpl_pose,
            smpl_params_weight = smpl_params_weight,
            index=torch.tensor(index)
        )
        return result
        

    def getitem(self, index):
        return self.__getitem__(index)

    def __len__(self):
        return len(self.data_list)

    @property
    def name(self):
        return 'Human36MDataset'

