from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp
import util.parallel_io as pio

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @property
    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass
    
    def load_annotation(self, anno_path):
        if osp.isdir(anno_path):
            all_data = pio.load_pkl_parallel(anno_path)
        else:
            all_data = pio.load_pkl_single(anno_path)
        
        if isinstance(all_data, list):
            return all_data
        else:
            data_list = list()
            for data in all_data.values():
                data_list.extend(data)
            return data_list
    

    def update_path(self, data_root, data_list):
        for data in data_list:
            for key in data.keys():
                if key.find("path")>=0:
                    data[key] = osp.join(data_root, data[key])