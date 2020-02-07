
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
from .loss_utils import LossUtil
import time
from util import vis_util


class DCTModel(BaseModel):
    @property
    def name(self):
        return 'DCTModel'

    def __init__(self, opt):

        BaseModel.initialize(self, opt)

        # set params
        self.inputSize = opt.inputSize

        self.single_branch = opt.single_branch
        self.two_branch = opt.two_branch
        self.aux_as_main = opt.aux_as_main
        assert (not self.single_branch and self.two_branch) or (
            self.single_branch and not self.two_branch)
        if self.aux_as_main:
            assert self.single_branch

        if opt.isTrain and opt.process_rank <= 0:
            if self.two_branch:
                print("!!!!!!!!!!!! Attention, use two branch framework")
                time.sleep(10)
            else:
                print("!!!!!!!!!!!! Attention, use one branch framework")
                if self.aux_as_main:
                    print("!!!!!!!!!!!! Attention, use IUV as input")
                    time.sleep(10)

        self.total_params_dim = opt.total_params_dim
        self.cam_params_dim = opt.cam_params_dim
        self.pose_params_dim = opt.pose_params_dim
        self.shape_params_dim = opt.shape_params_dim
        assert(self.total_params_dim ==
               self.cam_params_dim+self.pose_params_dim+self.shape_params_dim)
        self.dp_num_max = opt.dp_num_max

        if opt.dist:
            self.batchSize = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batchSize = opt.batchSize
        nb = self.batchSize

        # set input image and 2d keypoints
        self.input_img = self.Tensor(
            nb, opt.input_nc, self.inputSize, self.inputSize)
        self.input_IUV = self.Tensor(
            nb, opt.input_nc, self.inputSize, self.inputSize)
        self.IUV_weight = self.Tensor(nb, 1)
      
        self.keypoints = self.Tensor(nb, opt.keypoints_num, 2)
        self.keypoints_weights = self.Tensor(nb, opt.keypoints_num)
        # set 3d joints and smpl parameters
        self.joints_3d = self.Tensor(nb, opt.keypoints_num, 3)
        # Pose: 24*3, Shape: 10
        smpl_params_dim = opt.pose_params_dim + opt.shape_params_dim
        self.smpl_params = self.Tensor(nb, smpl_params_dim)
        self.smpl_params_weight = self.Tensor(nb, 1)


        # set densepose points
        batch_index = [np.ones((opt.dp_num_max*3,))*i for i in range(nb)]
        batch_index = np.array(batch_index).reshape(-1)
        self.batch_index = torch.from_numpy(batch_index).long().cuda()
        self.vert_index = torch.cuda.LongTensor(nb * opt.dp_num_max * 3)
        self.bc_coords = self.Tensor(nb * opt.dp_num_max)
        self.dp_keypoints = self.Tensor(nb, opt.dp_num_max, 2)
        self.dp_keypoints_weights = self.Tensor(nb, opt.dp_num_max)

        # since we use mixed batch, contains data with different annotations,
        # those weight is used to set data sample with different annotations, thus different loss weight
        self.dense_loss_weight = self.Tensor(nb, 1)
        self.joints_3d_loss_weight = self.Tensor(nb, 1)
        self.smpl_loss_weight = self.Tensor(nb, 1)
        self.has_dp_loss = False
        self.has_smpl_param_loss = False
        self.loss_util = LossUtil(opt)

        # load mean params, the mean params are from HMR
        self.mean_param_file = osp.join(
            opt.model_root, opt.mean_param_file)
        self.load_mean_params()

        # set differential SMPL (implemented with pytorch) and smpl_renderer
        self.smpl_face_path = osp.join(
            opt.model_root, opt.smpl_face_file)
        self.smpl_model_path = osp.join(
            opt.model_root, opt.smpl_model_file)
        self.smpl = SMPL(self.smpl_model_path, self.batchSize).cuda()

        # set encoder and optimizer
        self.encoder = dct_networks.DCTEncoder(opt).cuda()
        if opt.dist:
            self.encoder = DistributedDataParallel(
                self.encoder, device_ids=[torch.cuda.current_device()])
        if self.isTrain:
            self.optimizer_E = torch.optim.Adam(
                self.encoder.parameters(), lr=opt.lr_e)
        
        # load pretrained / trained weights for encoder
        if self.isTrain:
            if opt.continue_train:
                # resume training from saved weights
                which_epoch = opt.which_epoch
                saved_info = self.load_info(which_epoch)
                opt.epoch_count = saved_info['epoch']
                self.optimizer_E.load_state_dict(saved_info['optimizer_E'])
                which_epoch = opt.which_epoch
                self.load_network(self.encoder, 'encoder', which_epoch)
                if opt.process_rank <= 0:
                    print('resume from epoch {}'.format(opt.epoch_count))
            else:
                if opt.pretrained_weights is not None:
                    assert(osp.exists(opt.pretrained_weights))
                    if not self.opt.dist or self.opt.process_rank <= 0:
                        print("Load pretrained weights from {}".format(
                            opt.pretrained_weights))
                    if opt.dist:
                        self.encoder.module.load_state_dict(
                            torch.load(opt.pretrained_weights, 
                            map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())), 
                            strict=False)
                    else:
                        self.encoder.load_state_dict(
                            torch.load(opt.pretrained_weights))
        else:
            # load trained model for testing
            which_epoch = opt.which_epoch
            self.load_network(self.encoder, 'encoder', which_epoch)



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


    def set_input(self, input):
        input_img = input['img']
        self.input_img.resize_(input_img.size()).copy_(input_img)

        if 'IUV' in input:
            input_IUV = input['IUV']
            self.input_IUV.resize_(input_IUV.size()).copy_(input_IUV)

        keypoints = input['keypoints']
        keypoints_weights = input['keypoints_weights']
        self.keypoints.resize_(keypoints.size()).copy_(keypoints)
        self.keypoints_weights.resize_(
            keypoints_weights.size()).copy_(keypoints_weights)

        dense_loss_weight = input['dense_loss_weight']
        joints_3d_loss_weight = input['joints_3d_loss_weight']
        smpl_loss_weight = input['smpl_loss_weight']

        self.dense_loss_weight.resize_(
            dense_loss_weight.size()).copy_(dense_loss_weight)
        self.joints_3d_loss_weight.resize_(
            joints_3d_loss_weight.size()).copy_(joints_3d_loss_weight)
        self.smpl_loss_weight.resize_(
            smpl_loss_weight.size()).copy_(smpl_loss_weight)
        self.smpl_loss_weight.resize_(
            smpl_loss_weight.size()).copy_(smpl_loss_weight)

        self.has_dp_loss = False
        self.has_smpl_param_loss = False
        self.has_dp_vector_loss = False

        if 'dp_keypoints' in input:
            dp_keypoints = input['dp_keypoints']
            dp_keypoints_weights = input['dp_keypoints_weights']
            vert_index = input['vert_indices'].view(-1)
            bc_coords = input['bc_coords']
            self.dp_keypoints.resize_(dp_keypoints.size()).copy_(dp_keypoints)
            self.dp_keypoints_weights.resize_(
                dp_keypoints_weights.size()).copy_(dp_keypoints_weights)
            self.vert_index.resize_(vert_index.size()).copy_(vert_index)
            self.bc_coords.resize_(bc_coords.size()).copy_(bc_coords)
            self.has_dp_loss = True

        if 'smpl_shape' in input:
            smpl_shape = input['smpl_shape']
            smpl_pose = input['smpl_pose']
            smpl_params = torch.cat([smpl_pose, smpl_shape], dim=1)
            smpl_params_weight = input['smpl_params_weight']
            self.smpl_params.resize_(smpl_params.size()).copy_(smpl_params)
            self.smpl_params_weight.resize_(
                smpl_params_weight.size()).copy_(smpl_params_weight)
            self.has_smpl_param_loss = True


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

        # Since the gt global rotation is not trained,
        # During evaluation, setting the global rotation of predicted/GT smpl pose to both zero
        self.pred_global_rotation = self.pred_pose_params[:, :3].clone()
        if not self.opt.isTrain:
            self.pred_pose_params[:, :3] = 0.0
        self.pred_smpl_params = torch.cat(
            [self.pred_pose_params, self.pred_shape_params], dim=1)
        #  get predicted smpl verts, predict joints, and predict smpl joints
        self.pred_verts_3d, self.pred_kp_3d, _ = self.smpl(
            self.pred_shape_params, self.pred_pose_params)

        if not self.opt.isTrain:
            # for visualization and 2D prediction, use the predicted global rotation
            self.vis_pose_params = self.pred_pose_params.clone()
            self.vis_pose_params[:, :3] = self.pred_global_rotation.clone()
            self.pred_verts_vis, self.vis_kp_3d, _ = self.smpl(
                self.pred_shape_params, self.vis_pose_params)
            self.pred_verts_2d = batch_orth_proj_idrot(
                self.pred_verts_vis, self.pred_cam_params)
            self.pred_kp_2d = batch_orth_proj_idrot(
                self.vis_kp_3d, self.pred_cam_params)
        else:
            self.pred_verts_2d = batch_orth_proj_idrot(
                self.pred_verts_3d, self.pred_cam_params)
            self.pred_kp_2d = batch_orth_proj_idrot(
                self.pred_kp_3d, self.pred_cam_params)

        # get predicted densepose keypoints
        if self.has_dp_loss:
            selected_verts = self.pred_verts_2d[self.batch_index, self.vert_index, :].view(
                self.batchSize, self.dp_num_max*3, 2)
            weighted_verts = (
                selected_verts * self.bc_coords).view(self.batchSize, self.dp_num_max, 3, 2)
            self.pred_dp_keypoints = torch.sum(weighted_verts, dim=2)

        # In train phase, calculate gt_verst only when visulize images
        # In test phase, calculate gt_verts in every forward,
        if not self.opt.isTrain:
            gt_pose_params = self.smpl_params[:, :self.pose_params_dim]
            gt_pose_params[:, :3] = 0.0
            gt_shape_params = self.smpl_params[:, self.pose_params_dim:]
            self.gt_verts_3d, _, _ = self.smpl(gt_shape_params, gt_pose_params)

        # compute smpl joints
        if not self.opt.isTrain:
            zero_global_rotataion = torch.zeros((self.batchSize, 3)).float().cuda()
            # compute GT smpl joints (global rot set to be zero)
            gt_shape_params = self.smpl_params[:, self.pose_params_dim:]
            gt_joints_pose_params = torch.cat(
                (zero_global_rotataion, self.smpl_params[:, 3:self.pose_params_dim]), dim=1)
            _, _, self.smpl_joints = self.smpl(
                gt_shape_params, gt_joints_pose_params)
            # compute pred smpl joints (global rot set to be zero)
            pred_joints_pose_params = torch.cat(
                (zero_global_rotataion, self.pred_pose_params[:, 3:]), dim=1)
            _, _, self.pred_smpl_joints = self.smpl(
                self.pred_shape_params, pred_joints_pose_params)
        
        # compute T-Pose smpl verts
        if not self.opt.isTrain:
            tpose = torch.zeros(self.batchSize, 72).cuda()
            tpose[:, 0] = 3.39
            self.pred_verts_tpose = self.smpl(self.pred_shape_params, tpose)[0]
            gt_shape_params = self.smpl_params[:, self.pose_params_dim:]
            self.gt_verts_tpose = self.smpl(gt_shape_params, tpose)[0]


    def backward_E(self):
        # 2d keypoint loss
        self.keypoint_2d_loss = self.loss_util._keypoint_2d_loss(
            self.keypoints, self.pred_kp_2d, self.keypoints_weights)
        self.keypoint_2d_loss *= self.opt.kp_loss_weight
        self.loss = self.keypoint_2d_loss

        # densepose alignment loss
        if self.has_dp_loss:
            self.dp_align_loss = self.loss_util._densepose_align_loss(
                self.dp_keypoints, self.pred_dp_keypoints, self.dp_keypoints_weights, self.dense_loss_weight)
            self.dp_align_loss *= self.opt.dp_align_loss_weight
            self.loss = (self.loss + self.dp_align_loss)

        # smpl params loss
        if self.has_smpl_param_loss:
            self.smpl_params_loss = self.loss_util._smpl_params_loss(
                self.smpl_params, self.pred_smpl_params, self.smpl_params_weight, self.smpl_loss_weight)
            self.smpl_params_loss *= self.opt.loss_3d_weight
            self.loss = self.loss + self.smpl_params_loss
            # smpl joints loss, similar with 3D joints loss

        self.loss.backward()


    def optimize_parameters(self):
        self.optimizer_E.zero_grad()
        self.backward_E()
        self.optimizer_E.step()


    def compute_loss(self):
        # 2d keypoint loss
        self.keypoint_2d_loss = self.loss_util._keypoint_2d_loss(
            self.keypoints, self.pred_kp_2d, self.keypoints_weights)
        res_dict = OrderedDict(
            [('kp_loss', self.keypoint_2d_loss.detach().cpu().numpy())])

        # densepose alignment loss
        if self.has_dp_loss:
            self.dp_align_loss = self.loss_util._densepose_align_loss(
                self.dp_keypoints, self.pred_dp_keypoints, self.dp_keypoints_weights, self.dense_loss_weight)
            res_dict['dp_loss'] = self.dp_align_loss.detach().cpu().numpy()

        # smpl params loss and potential smpl joints loss
        if self.has_smpl_param_loss:
            self.smpl_params_loss = self.loss_util._smpl_params_loss(
                self.smpl_params, self.pred_smpl_params, self.smpl_params_weight, self.smpl_loss_weight)
            res_dict['smpl_loss'] = self.smpl_params_loss.detach().cpu().numpy()
         
        return res_dict


    def test(self):
        with torch.no_grad():
            self.forward()


    def get_pred_result(self):
        pred_result = OrderedDict({
            'cams': self.pred_cam_params.cpu().numpy(),
            'shape_params': self.pred_shape_params.cpu().numpy(),
            'pose_params': self.pred_pose_params.cpu().numpy(),
            'pred_verts': self.pred_verts_3d.cpu().numpy(),
            'gt_verts': self.gt_verts_3d.cpu().numpy(),
            'pred_verts_vis': self.pred_verts_vis.cpu().numpy(),
            'smpl_joints': self.smpl_joints.cpu().numpy(),
            'pred_smpl_joints': self.pred_smpl_joints.cpu().numpy(),
            'gt_verts_tpose': self.gt_verts_tpose.cpu().numpy(),
            'pred_verts_tpose': self.pred_verts_tpose.cpu().numpy()
        })
        return pred_result


    def get_current_errors(self):
        kp_loss = self.keypoint_2d_loss.item()
        total_loss = self.loss.item()
        loss_dict = OrderedDict([('kp_loss', kp_loss)])

        if self.has_dp_loss:
            dp_align_loss = self.dp_align_loss.item()
            loss_dict['dp_align_loss'] = dp_align_loss

        if self.has_smpl_param_loss:
            smpl_params_loss = self.smpl_params_loss.item()
            loss_dict['smpl_params_loss'] = smpl_params_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict


    def rend_img(self, img, vert, cam):
        f = 5.
        tz = f / cam[0]
        cam_for_render = 0.5 * self.inputSize * np.array([f, 1, 1])
        cam_t = np.array([cam[1], cam[2], tz])
        rend_img = vis_util.render_image(
            vert+cam_t, cam_for_render, img, self.opt.inputSize, self.smpl_face_path)
        return rend_img


    def get_current_visuals(self, idx=0):
        # visualize image and IUV first
        img = self.input_img[idx].cpu().detach().numpy()
        show_img = vis_util.recover_img(img)[:,:,::-1]
        IUV = self.input_IUV[idx].cpu().detach().numpy()
        IUV = vis_util.recover_IUV(IUV)
        visual_dict = OrderedDict([('img', show_img), ('IUV', IUV)])

        # visualize keypoint
        kp = self.keypoints[idx].cpu().detach().numpy()
        pred_kp = self.pred_kp_2d[idx].cpu().detach().numpy()
        kp_weight = self.keypoints_weights[idx].cpu().detach().numpy()
        kp_img = vis_util.draw_keypoints(
            img, kp, kp_weight, 'red', self.inputSize)
        pred_kp_img = vis_util.draw_keypoints(
            img, pred_kp, kp_weight, 'green', self.inputSize)

        # visualize image with 2D joints and rendered SMPL
        if not self.opt.isTrain and self.has_smpl_param_loss:
            vert = self.pred_verts_vis[idx].cpu().detach().numpy()
        else:
            vert = self.pred_verts_3d[idx].cpu().detach().numpy()
        cam = self.pred_cam_params[idx].cpu().detach().numpy()
        rend_img = self.rend_img(img, vert, cam)
        rend_img = rend_img[:, :, ::-1]
        visual_dict['rend_img'] = rend_img
        visual_dict['gt_keypoint'] = kp_img
        visual_dict['pred_keypoint'] = pred_kp_img

        # visualize dense keypoints
        if self.has_dp_loss:
            dp_kp = self.dp_keypoints[idx].cpu().detach().numpy()
            pred_dp_kp = self.pred_dp_keypoints[idx].cpu().detach().numpy()
            dp_kp_weight = self.dp_keypoints_weights[idx].cpu(
            ).detach().numpy()
            dp_kp_img = vis_util.draw_dp_anno(
                img, dp_kp, dp_kp_weight, 'red', self.inputSize)
            pred_dp_kp_img = vis_util.draw_dp_anno(
                img, pred_dp_kp, dp_kp_weight, 'green', self.inputSize)
            visual_dict['gt_dense_keypoint'] = dp_kp_img
            visual_dict['pred_dense_keypoint'] = pred_dp_kp_img

        return visual_dict


    def get_current_visuals_batch(self):
        all_visuals = list()
        for idx in range(self.batchSize):
            all_visuals.append(self.get_current_visuals(idx))
        return all_visuals


    def save(self, label, epoch):
        self.save_network(self.encoder, 'encoder', label)
        save_info = {'epoch': epoch,
                     'optimizer_E': self.optimizer_E.state_dict()}
        self.save_info(save_info, label)


    def eval(self):
        self.encoder.eval()


    def update_learning_rate(self, epoch):
        old_lr = self.opt.lr_e
        lr = 0.5*(1.0 + np.cos(np.pi*epoch/self.opt.total_epoch)) * old_lr
        for param_group in self.optimizer_E.param_groups:
            param_group['lr'] = lr
        if self.opt.process_rank <= 0:
            print("Current Learning Rate:{0:.2E}".format(lr))
