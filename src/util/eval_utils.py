from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import sys


def pad_and_resize(img, final_size=224):
    height, width = img.shape[:2]
    new_len = np.max([height, width])
    new_img = np.zeros((new_len, new_len, 3), dtype=np.uint8)
    new_IUV = np.zeros((new_len, new_len, 3), dtype=np.uint8)
    if height > width:
        margin = (height-width)//2
        new_img[:, margin:margin+width, :] = img
    elif height < width:
        margin = (width-height)//2
        new_img[margin:margin+height, :, :] = img
    else:
        new_img[:, :, :] = img
    new_img = cv2.resize(new_img, (final_size, final_size))
    return new_img


def get_subdir(img_path):
    record = img_path.split('/')[:-1]
    for i in range(len(record)):
        if record[i].find('image')>=0:
            subdir = '/'.join(record[i+1:])
            return subdir