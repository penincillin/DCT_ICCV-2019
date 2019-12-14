from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dist', action='store_true', help='whether to use distributed training')
        self.parser.add_argument('--local_rank', type=int, default=0)
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--inputSize', type=int, default=224, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--main_encoder', type=str, default='resnet50', help='selects model to use for major input, it is usually image')
        self.parser.add_argument('--aux_encoder', type=str, default='resnet18', help='selects model to use for auxiliary input, it could be IUV') 
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='dp2smpl', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=80, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=1, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        self.parser.add_argument('--data_root', type=str, default='', help='root dir for all the datasets')
        self.parser.add_argument('--coco_anno_path', type=str, default='', help='annotation_path that stores the information of coco dataset')
        self.parser.add_argument('--human36m_anno_path', type=str, default='', help='annotation_path that stores the information of human36m dataset')
        self.parser.add_argument('--up3d_anno_path', type=str, default='', help='annotation_path that stores the information of up3d dataset')
        self.parser.add_argument('--up3d_use3d', action='store_true', help='use 3d data of up3d dataset')

        self.parser.add_argument('--dp_num_max', type=int, default=400, help='max number of densepose annotations')
        self.parser.add_argument('--refine_IUV', action='store_true', help='use refined IUV map and dense points')
        self.parser.add_argument('--keypoints_num', type=int, default=19, help='number of keypoints')
        self.parser.add_argument('--total_params_dim', type=int, default=85, help='number of params to be estimated')
        self.parser.add_argument('--cam_params_dim', type=int, default=3, help='number of params to be estimated')
        self.parser.add_argument('--pose_params_dim', type=int, default=72, help='number of params to be estimated')
        self.parser.add_argument('--shape_params_dim', type=int, default=10, help='number of params to be estimated')

        self.parser.add_argument('--model_root', type=str, default='', help='root dir for all the pretrained weights and pre-defined models')
        self.parser.add_argument('--smpl_model_file', type=str, default='smpl_cocoplus_neutral_no_chumpy.pkl', help='path of pretraind smpl model')
        self.parser.add_argument('--smpl_face_file', type=str, default='smpl_faces.npy', help='path of smpl face')
        self.parser.add_argument('--mean_param_file', type=str, default='neutral_smpl_mean_params.h5', help='path of smpl face')

        self.parser.add_argument('--single_branch', action='store_true', help='use only one branch, this branch could either be IUV or other format such as image')
        self.parser.add_argument('--two_branch', action='store_true', help='two branch input, image and another auxiliary branch, the auxiliary branch is IUV in default')
        self.parser.add_argument('--aux_as_main', action='store_true', help='use aux as input instead of image')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')


        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt
