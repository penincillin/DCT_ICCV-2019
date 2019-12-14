
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# dct is the abbr. of Human Model recovery with Densepose supervision
import numpy as np
import torch
import os
import sys
import shutil
import os.path as osp
from collections import OrderedDict
import deepdish
import itertools
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import pdb
import cv2
from .base_model import BaseModel
from . import resnet
from . import dct_networks
from .smpl import SMPL, batch_orth_proj_idrot, batch_rodrigues
import time


class DCTModel(BaseModel):
    @property
    def name(self):
        return 'DCTModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)
        # set params
        self.inputSize = opt.inputSize
        if opt.dist:
            self.batchSize = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batchSize = opt.batchSize
        nb = self.batchSize

        # set architecture of the model
        self.set_model_arch()
        # initialize input of the model
        self.init_input()
        # intialize SMPL related
        self.init_smpl()
        # set encoder and load trained model for testing
        self.encoder = dct_networks.DCTEncoder(opt).cuda()
        if opt.dist:
            self.encoder = DistributedDataParallel(
                self.encoder, device_ids=[torch.cuda.current_device()])
        self.load_network(self.encoder, opt.trained_model_path)


    def set_model_arch(self):
        self.single_branch = self.opt.single_branch
        self.two_branch = self.opt.two_branch
        self.aux_as_main = self.opt.aux_as_main
        assert (not self.single_branch and self.two_branch) or (
            self.single_branch and not self.two_branch)
        if self.aux_as_main:
            assert self.single_branch
        if self.opt.isTrain and self.opt.process_rank <= 0:
            if self.two_branch:
                print("!!!!!!!!!!!! Attention, use two branch framework")
                time.sleep(10)
            else:
                print("!!!!!!!!!!!! Attention, use one branch framework")


    def init_input(self):
        self.dp_num_max = self.opt.dp_num_max
        nb = self.batchSize
        self.input_img = self.Tensor(
            nb, self.opt.input_nc, self.inputSize, self.inputSize)
        self.input_IUV = self.Tensor(
            nb, self.opt.input_nc, self.inputSize, self.inputSize)


    def set_input(self, input):
        input_img = input['img']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        if 'IUV' in input:
            input_IUV = input['IUV']
            self.input_IUV.resize_(input_IUV.size()).copy_(input_IUV)


    def init_smpl(self):
        # dim of SMPL params
        self.total_params_dim = self.opt.total_params_dim
        self.cam_params_dim = self.opt.cam_params_dim
        self.pose_params_dim = self.opt.pose_params_dim
        self.shape_params_dim = self.opt.shape_params_dim
        assert(self.total_params_dim ==
               self.cam_params_dim+self.pose_params_dim+self.shape_params_dim)
        # load mean params, the mean params are from HMR
        self.mean_param_file = osp.join(
            self.opt.model_root, 'neutral_smpl_mean_params.h5')
        self.load_mean_params()
        # set differential SMPL (implemented with pytorch) and smpl_renderer
        self.smpl_face_path = osp.join(
            self.opt.model_root, self.opt.smpl_face_file)
        self.smpl_model_path = osp.join(
            self.opt.model_root, self.opt.smpl_model_file)
        self.smpl = SMPL(self.smpl_model_path, self.batchSize).cuda()
        if self.opt.dist:
            self.smpl = DistributedDataParallel(
                self.smpl, device_ids=[torch.cuda.current_device()])


    def load_mean_params(self):
        mean_params = np.zeros((1, self.total_params_dim))
        mean_vals = deepdish.io.load(self.mean_param_file)
        # Initialize scale at 0.9
        mean_params[0, 0] = 0.9
        # set pose
        mean_pose = mean_vals['pose']
        mean_pose[:3] = 0.
        mean_pose[0] = np.pi
        # set shape
        mean_shape = mean_vals['shape']
        mean_params[0, 3:] = np.hstack((mean_pose, mean_shape))
        mean_params = np.repeat(mean_params, self.batchSize, axis=0)
        self.mean_params = torch.from_numpy(mean_params).float().cuda()
        self.mean_params.requires_grad = False


    def forward(self):
        cam_dim = self.cam_params_dim
        pose_dim = self.pose_params_dim
        shape_dim = self.shape_params_dim

        if not self.aux_as_main:
            self.output = self.encoder(self.input_img, self.input_IUV)
        else:
            self.output = self.encoder(self.input_IUV, self.input_IUV)
        self.final_params = self.output + self.mean_params

        # get predicted params for cam, pose, shape
        self.pred_cam_params = self.final_params[:, :cam_dim]
        self.pred_pose_params = self.final_params[:, cam_dim: (
            cam_dim + pose_dim)]
        self.pred_shape_params = self.final_params[:, (cam_dim + pose_dim):]

        # get predicted smpl verts, predict joints, and rotation matrix
        self.pred_verts_3d, self.pred_kp_3d, _ = self.smpl(
            self.pred_shape_params, self.pred_pose_params)

        self.pred_verts_2d = batch_orth_proj_idrot(
            self.pred_verts_3d, self.pred_cam_params)
        self.pred_kp_2d = batch_orth_proj_idrot(
            self.pred_kp_3d, self.pred_cam_params)


    def test(self):
        with torch.no_grad():
            self.forward()


    def get_pred_result(self):
        pred_result = OrderedDict({
            'cams': self.pred_cam_params.cpu().numpy(),
            'shape_params': self.pred_shape_params.cpu().numpy(),
            'pose_params': self.pred_pose_params.cpu().numpy(),
            'pred_verts': self.pred_verts_3d.cpu().numpy(),
        })
        return pred_result


    def eval(self):
        self.encoder.eval()