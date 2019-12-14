from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
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
from data.human36m_dataset import Human36MDataset
from data.coco_dataset import COCODataset
from data.up3d_dataset import UP3DDataset
import numpy as np


class ComposeDataset(BaseDataset):

    def __init__(self, opt):
        self.opt = opt
        candidate_datasets = list()

        # coco dataset
        if opt.isTrain and opt.train_coco:
            coco_dataset = COCODataset(opt)
            candidate_datasets.append(coco_dataset)

        # up3d dataset
        if (opt.isTrain and opt.train_up3d ) or \
             (not opt.isTrain and opt.test_dataset == 'up3d'):
            up3d_dataset = UP3DDataset(opt)
            candidate_datasets.append(up3d_dataset)

        # human36m dataset
        if opt.isTrain:
            # by default, the number of human3.6M data used in training is set to be
            # # same as the sum of number of other datasets
            h36m_data_num = int(np.sum([len(dataset) for dataset in candidate_datasets]))
            human36m_dataset = Human36MDataset(opt, h36m_data_num)
            candidate_datasets.append(human36m_dataset)
        
        assert(len(candidate_datasets)>0)
        if opt.process_rank <= 0 and opt.isTrain:
            for dataset in candidate_datasets:
                print('{} dataset has {} data'.format(dataset.name, len(dataset)))

        self.all_datasets = list()
        for dataset in candidate_datasets:
            if len(dataset) > 0:
                self.all_datasets.append(dataset)
        self.index_map = self.get_index_map()
        self.size_dict = self.get_size_dict()


    def get_size_dict(self):
        size_dict = dict()
        for dataset in self.all_datasets:
            data = dataset.getitem(0)
            for key in data.keys():
                size_dict[key] = data[key].size()
        return size_dict

    def get_index_map(self):
        total_data_num = self.__len__()
        index_map = list()
        for dataset_id, dataset in enumerate(self.all_datasets):
            dataset_len = len(dataset)
            index_map += [(dataset_id, idx) for idx in range(dataset_len)]
        return index_map

    def complete_data(self, data):
        dense_weight = 0
        joints_3d_weight = 0
        smpl_weight = 0
        if 'dp_keypoints' in data:
            dense_weight = 1
        if 'smpl_pose' in data:
            smpl_weight = 1
        if 'joints_3d' in data:
            joints_3d_weight = 1

        result_data = dict()
        for key in self.size_dict:
            if key in data:
                result_data[key] = data[key]
            else:
                if key == 'vert_indices':
                    add_data = torch.zeros(self.size_dict[key]).long()
                else:
                    add_data = torch.zeros(self.size_dict[key]).float()
                result_data[key] = add_data

        result_data['dense_loss_weight'] = \
            torch.from_numpy(np.array([dense_weight, ])).float()
        result_data['joints_3d_loss_weight'] = \
            torch.from_numpy(np.array([joints_3d_weight, ])).float()
        result_data['smpl_loss_weight'] = \
            torch.from_numpy(np.array([smpl_weight, ])).float()

        return result_data

    def __getitem__(self, index):
        dataset_id, dataset_index = self.index_map[index]
        dataset = self.all_datasets[dataset_id]
        data = dataset.getitem(dataset_index)
        res_data = self.complete_data(data)
        return res_data


    def shuffle_data(self):
        for dataset in self.all_datasets:
            random.shuffle(dataset.data_list)


    def __len__(self):
        total_data_num = sum([len(dataset) for dataset in self.all_datasets])
        return total_data_num

    @property
    def name(self):
        'ComposeDataset'
