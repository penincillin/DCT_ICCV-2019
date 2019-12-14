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
import cv2
import matplotlib.pyplot as plt
import pickle


class DataProcessor(object):

    def __init__(self, opt):
        self.opt = opt
        self.dp_num_max = opt.dp_num_max


    def padding_and_resize(self, img, kps, IUV=None, dp_kps=None):
        height, width = img.shape[:2]

        new_len = np.max([height, width])
        delta_x = (new_len-width)//2
        delta_y = (new_len-height)//2
        delta = np.array([delta_x, delta_y, 0]).reshape(1, 3)

        new_kps = np.copy(kps)
        # add shift
        new_kps += np.repeat(delta, new_kps.shape[0], axis=0)
        kps_weight = np.ones(kps.shape[0])
        kps_weight[kps[:, 2] == 0] = 0
        visible_ky_num = np.count_nonzero(kps_weight[:-1])
        kps_weight *= (kps.shape[0] / visible_ky_num)
        kps_weight = np.stack([kps_weight, kps_weight], axis=1)

        if dp_kps is not None:
            new_dp_kps = np.zeros((self.dp_num_max, 2))
            new_dp_kps[:dp_kps.shape[0], :] = dp_kps
            new_dp_kps += np.repeat(delta[:, :2], new_dp_kps.shape[0], axis=0)
            dp_kps_weight = np.ones(self.dp_num_max)
            dp_kps_weight[dp_kps.shape[0]:] = 0
            visible_dp_kp_num = np.count_nonzero(dp_kps_weight[:-1])
            dp_kps_weight *= (self.dp_num_max / visible_dp_kp_num)
            dp_kps_weight = np.stack([dp_kps_weight, dp_kps_weight], axis=1)
        else:
            new_dp_kps = None
            dp_kps_weight = None

        # resize
        new_img = np.zeros((new_len, new_len, 3), dtype=np.uint8)
        new_IUV = np.zeros((new_len, new_len, 3), dtype=np.uint8)
        if IUV is None:
            IUV = np.zeros(img.shape)

        if height > width:
            margin = (height-width)//2
            new_img[:, margin:margin+width, :] = img
            new_IUV[:, margin:margin+width, :] = IUV
        elif height < width:
            margin = (width-height)//2
            new_img[margin:margin+height, :, :] = img
            new_IUV[margin:margin+height, :, :] = IUV
        else:
            new_img[:, :, :] = img
            new_IUV[:, :, :] = IUV


        finalSize = self.opt.inputSize
        x_scale = finalSize/new_len
        y_scale = finalSize/new_len
        new_img = cv2.resize(new_img, (finalSize, finalSize))
        new_IUV = cv2.resize(
            new_IUV, (finalSize, finalSize), interpolation=cv2.INTER_NEAREST)

        scale_ratio = np.array([x_scale, y_scale, 1]).reshape(1, 3)
        new_kps *= np.repeat(scale_ratio, new_kps.shape[0], axis=0)
        # remove the visibilty
        new_kps = new_kps[:, :2]

        if new_dp_kps is not None:
            new_dp_kps *= np.repeat(scale_ratio[:, :2],
                                    new_dp_kps.shape[0], axis=0)

        return new_img, new_kps, kps_weight, new_IUV, new_dp_kps, dp_kps_weight

    def flip_IUV(self, IUV):
        flipped_IUV = np.zeros(IUV.shape, dtype=np.uint8)
        height, width = flipped_IUV.shape[:2]

        # set background
        background_ids = [0, ]
        for part_idx in background_ids:
            y, x = np.where(IUV[:, :, 0] == part_idx)
            f_y, f_x = y, width-x-1
            flipped_IUV[f_y, f_x, 0] = IUV[y, x, 0]
            flipped_IUV[f_y, f_x, 1] = IUV[y, x, 1]
            flipped_IUV[f_y, f_x, 2] = IUV[y, x, 2]

        # set torso
        torso_idxs = [1, 2]
        for part_idx in torso_idxs:
            y, x = np.where(IUV[:, :, 0] == part_idx)
            f_y, f_x = y, width-x-1
            flipped_IUV[f_y, f_x, 0] = IUV[y, x, 0]
            flipped_IUV[f_y, f_x, 1] = IUV[y, x, 1]
            flipped_IUV[f_y, f_x, 2] = 255-IUV[y, x, 2]

        # set hand, foot, head
        symmetric_idxs = [3, 4, 5, 6, 23, 24]
        for part_idx in symmetric_idxs:
            y, x = np.where(IUV[:, :, 0] == part_idx)
            f_y, f_x = y, width-x-1
            new_part_idx = part_idx + (1 if part_idx % 2 == 1 else -1)
            flipped_IUV[f_y, f_x, 0] = new_part_idx
            flipped_IUV[f_y, f_x, 1] = 255-IUV[y, x, 1]
            flipped_IUV[f_y, f_x, 2] = IUV[y, x, 2]

        # set legs and arms
        non_symmetric_idxs = range(7, 23)
        for part_idx in non_symmetric_idxs:
            y, x = np.where(IUV[:, :, 0] == part_idx)
            f_y, f_x = y, width-x-1
            new_part_idx = part_idx + (1 if part_idx % 2 == 1 else -1)
            flipped_IUV[f_y, f_x, 0] = new_part_idx
            flipped_IUV[f_y, f_x, 1] = IUV[y, x, 1]
            flipped_IUV[f_y, f_x, 2] = IUV[y, x, 2]

        return flipped_IUV

    def flip_joints_3d(self, joints_3d):
        swap_inds = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
        joints_ref = joints_3d[swap_inds, :]
        flip_mat = np.array(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        joints_ref = np.transpose(
            np.matmul(flip_mat, np.transpose(joints_ref)))
        joints_ref = joints_ref - np.mean(joints_ref, axis=0)
        return joints_ref

    def flip_smpl_pose(self, smpl_pose):
        """ {{{ # How I got the indices:
        right = [11, 8, 5, 2, 14, 17, 19, 21, 23]
        left = [10, 7, 4, 1, 13, 16, 18, 20, 22]
        new_map = {}
        for r_id, l_id in zip(right, left):
            for axis in range(0, 3):
                rind = r_id * 3 + axis
                lind = l_id * 3 + axis
                new_map[rind] = lind
                new_map[lind] = rind
        asis = [id for id in np.arange(0, 24) if id not in right + left]
        for a_id in asis:
            for axis in range(0, 3):
                aind = a_id * 3 + axis
                new_map[aind] = aind
        swap_inds = np.array([new_map[k] for k in sorted(new_map.keys())])
        }}} """
        swap_inds = [
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
            19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
            36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
            50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
            67, 68]
        sign_flip = [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                     -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
                     -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
                     1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                     -1, 1, -1, -1]
        new_smpl_pose = smpl_pose[swap_inds] * sign_flip
        return new_smpl_pose


    def random_flip(self, img, kps, kps_weight, IUV,
                    dp_kps, dp_kps_weight, joints_3d, smpl_pose):

        if np.random.random() < 0.5:
            return img, kps, kps_weight, IUV, \
                dp_kps, dp_kps_weight, joints_3d, smpl_pose, False

        else:
            height, width = img.shape[:2]
            # flip img and IUV
            new_img = np.fliplr(img).copy()
            if IUV is not None:
                new_IUV = self.flip_IUV(IUV)
            else:
                new_IUV = None
                
            # flip 2d keypoints
            swap_inds = [5, 4, 3, 2, 1, 0, 11, 10, 9,
                         8, 7, 6, 12, 13, 14, 16, 15, 18, 17]
            new_kps = kps[swap_inds, :]
            new_kps_weight = kps_weight[swap_inds, :]
            for i in range(new_kps.shape[0]):
                new_x = (width-1) - new_kps[i, 0]
                new_kps[i, 0] = new_x

            # flip densepose keypoints
            new_dp_kps = dp_kps
            new_dp_kps_weight = dp_kps_weight
            if dp_kps is not None and dp_kps_weight is not None:
                for i in range(new_dp_kps.shape[0]):
                    new_x = (width-1) - new_dp_kps[i, 0]
                    new_dp_kps[i, 0] = new_x

            # flip 3d joints and smpl_pose
            if joints_3d is not None:
                new_joints_3d = self.flip_joints_3d(joints_3d)
            else:
                new_joints_3d = None
            if smpl_pose is not None:
                new_smpl_pose = self.flip_smpl_pose(smpl_pose)
            else:
                new_smpl_pose = None

            return new_img, new_kps, new_kps_weight, \
                new_IUV, new_dp_kps, new_dp_kps_weight, \
                new_joints_3d, new_smpl_pose, True


    def normalize_keypoints(self, keypoints, dp_keypoints):
        finalSize = self.opt.inputSize

        new_kps = np.copy(keypoints)
        new_kps[:, 0] = (keypoints[:, 0] / finalSize) * 2.0 - 1.0
        new_kps[:, 1] = (keypoints[:, 1] / finalSize) * 2.0 - 1.0

        new_dp_kps = np.copy(dp_keypoints)
        if dp_keypoints is not None:
            new_dp_kps[:, 0] = (dp_keypoints[:, 0] / finalSize) * 2.0 - 1.0
            new_dp_kps[:, 1] = (dp_keypoints[:, 1] / finalSize) * 2.0 - 1.0
        return new_kps, new_dp_kps



    def transform_IUV(self, IUV):
        IUV_tensor = torch.from_numpy(IUV).float()
        IUV_tensor = IUV_tensor.permute(2, 0, 1)

        IUV_tensor[0] = (IUV_tensor[0]/24 - 0.5) * 2
        IUV_tensor[1] = (IUV_tensor[1]/255 - 0.5) * 2
        IUV_tensor[2] = (IUV_tensor[2]/255 - 0.5) * 2
        return IUV_tensor
    

    def refine_dp_kps(self, IUV, dp_kps, dp_kps_weight):
        height, width = IUV.shape[:2]
        num_valid = 0
        for i, (x, y) in enumerate(dp_kps):
            x, y = int(x), int(y)
            x = min(max(x, 0), width-1)
            y = min(max(y, 0), height-1)
            if IUV[y, x, 0] > 0:
                num_valid += 1
            else:
                dp_kps_weight[i] = 0.0
        if num_valid > 0:
            dp_kps_weight *= (1.0 * dp_kps.shape[0] / num_valid)
        else:
            dp_kps_weight *= 0.0
        return dp_kps_weight