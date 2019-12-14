
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import shutil
import os.path as osp
from collections import OrderedDict
import deepdish
import itertools
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb
import cv2
from .smpl import batch_rodrigues


class LossUtil(object):

    def __init__(self, opt):
        self.inputSize = opt.inputSize
        self.pose_params_dim = opt.pose_params_dim
        self.isTrain = opt.isTrain
        if opt.dist:
            self.batchSize = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batchSize = opt.batchSize


    def _keypoint_2d_loss(self, target_keypoint, pred_keypoint, keypoint_weights):
        abs_loss = torch.abs((target_keypoint-pred_keypoint))
        weighted_loss = abs_loss * keypoint_weights
        if self.isTrain:
            loss = torch.mean(weighted_loss)
        else:
            loss = weighted_loss
        return loss


    def _densepose_align_loss(self, dp_keypoints, pred_dp_keypoints, dp_keypoints_weights, loss_weight):
        # calcluate L1 loss
        abs_loss = torch.abs(pred_dp_keypoints - dp_keypoints)

        # calc wegithed loss
        loss_weight = loss_weight.view(self.batchSize, 1, 1)
        weighted_loss = abs_loss * dp_keypoints_weights * loss_weight
        # 1e-8 in case that torch.sum(loss_weight) == 0
        rescale_rate = self.batchSize / (torch.sum(loss_weight) + 1e-8)

        if self.isTrain:
            loss = torch.mean(weighted_loss) * rescale_rate
        else:
            loss = weighted_loss * rescale_rate
        return loss


    def _smpl_params_loss(self, smpl_params, pred_smpl_params, smpl_params_weight, loss_weight):
        # change pose parameters to rodrigues matrix
        # pose_params shape (bs, 72), pose_rodrigues shape(bs, 24, 3, 3)
        pose_params = smpl_params[:, :self.pose_params_dim].contiguous()
        shape_params = smpl_params[:, self.pose_params_dim:]
        pose_rodrigues = batch_rodrigues(pose_params.view(-1, 3)).view(self.batchSize, 24, 3, 3)

        pred_pose_params = pred_smpl_params[:, :self.pose_params_dim].contiguous()
        pred_shape_params = pred_smpl_params[:, self.pose_params_dim:]
        pred_pose_rodrigues = batch_rodrigues(pred_pose_params.view(-1, 3)).view(\
            self.batchSize, 24, 3, 3)
        
        # neglect the global rotation, the first 9  elements of rodrigues matrix
        pose_params = pose_rodrigues[:, 1:, :, :].view(self.batchSize, -1)
        pred_pose_params = pred_pose_rodrigues[:, 1:, :, :].view(self.batchSize, -1)

        smpl_params = torch.cat(
            [pose_params, smpl_params[:, self.pose_params_dim:]], dim=1)
        pred_smpl_params = torch.cat(
            [pred_pose_params, pred_smpl_params[:, self.pose_params_dim:]], dim=1)

        # square loss
        params_diff = smpl_params - pred_smpl_params
        square_loss = torch.mul(params_diff, params_diff)
        square_loss = square_loss * smpl_params_weight

        # calc weighted loss
        loss_weight = loss_weight.view(self.batchSize, 1)
        weighted_loss = square_loss * loss_weight 
        # 1e-8 in case that torch.sum(loss_weight) == 0
        rescale_rate = self.batchSize / (torch.sum(loss_weight) + 1e-8)

        if self.isTrain:
            loss = torch.mean(weighted_loss) * rescale_rate
        else:
            loss = weighted_loss * rescale_rate
        return loss