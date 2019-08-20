from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
from PIL import Image
from scipy import misc
import cv2
import pickle
from PIL import Image, ImageDraw
import util.parallel_io as pio
import util.ry_utils as ry_utils

def load_from_flie_list(file_path):
    all_data = list()
    with open(file_path, 'r') as in_f:
        for line in in_f:
            single_data = dict(
                image_path=line.strip()
            )
            all_data.append(single_data)
    return all_data


def load_from_annotation(anno_path):
    if osp.isdir(anno_path):
        all_data = pio.load_pkl_parallel(anno_path)
    elif anno_path.endswith('.pkl'):
        all_data = pio.load_pkl_single(anno_path)
    elif anno_path.endswith('.txt'):
        all_data = load_from_flie_list(anno_path)
    else:
        sys.exit(0)
        print("Unsupported Data Format")
    
    if isinstance(all_data, list):
        return all_data
    else:
        data_list = list()
        for data in all_data.values():
            data_list.extend(data)
        return data_list


def draw_keypoints(image, kps, weight, img_path):
    im = Image.fromarray(image[:,:,::-1])
    draw = ImageDraw.Draw(im)
    for i in range(kps.shape[0]):
        x, y = kps[i,:]
        if weight[i][0] > 0:
            draw.ellipse((x-2,y-2,x+2,y+2), fill='red')
    del draw
    im.save(img_path)


def draw_dp_anno(image, dp_kps, weight, img_path):
    im = Image.fromarray(image[:,:,::-1])
    draw = ImageDraw.Draw(im)
    for i in range(dp_kps.shape[0]):
        x, y = dp_kps[i,:]
        if weight[i][0]>0:
            draw.ellipse((x-2,y-2,x+2,y+2), fill='green')
    del draw
    im.save(img_path)


def render_img(img, vert, cam, renderer):
    f = 5.
    tz = f / cam[0]
    inputSize = 224
    cam_for_render = 0.5 * inputSize * np.array([f, 1, 1])
    cam_t = np.array([cam[1], cam[2], tz])
    # Undo pre-processing.
    input_img = img/255.0
    rend_img = renderer(vert + cam_t, cam_for_render, img=input_img)
    return rend_img