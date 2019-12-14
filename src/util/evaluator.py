from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import os.path as osp
import numpy as np
import pickle
import copy
import cv2
import time
import util.parallel_io as pio
import util.ry_utils as ry_utils
import util.eval_utils as eval_utils
import multiprocessing as mp


class Evaluator(object):

    def __init__(self, data_list=None, model_root=''):
        if data_list is not None:
            self.data_list = copy.deepcopy(data_list)
        if len(model_root) > 0:
            self.model_root = model_root
        self.pred_results = list()

    def clear(self):
        self.pred_results = list()

    def update(self, data_idxs, losses, pred_results):
        for i, data_idx in enumerate(data_idxs):
            single_data = dict(
                data_idx=data_idx,
                cam=pred_results['cams'][i],
                smpl_shape=pred_results['shape_params'][i],
                smpl_pose=pred_results['pose_params'][i],
                gt_smpl_shape=self.data_list[data_idx]['smpl_shape'],
                gt_smpl_pose=self.data_list[data_idx]['smpl_pose'],
                vis_verts=pred_results['pred_verts_vis'][i].astype(np.float16)
            )
            # pve
            verts1 = pred_results['pred_verts'][i]
            verts2 = pred_results['gt_verts'][i]
            single_data['pve'] = np.average(
                np.linalg.norm(verts1-verts2, axis=1))
            # mpjpe            
            smpl_joints1 = pred_results['smpl_joints'][i]
            smpl_joints2 = pred_results['pred_smpl_joints'][i]
            single_data['mpjpe'] = np.average(
                np.linalg.norm(smpl_joints1-smpl_joints2, axis=1))
            # pve_tpose
            pve_tpose1 = pred_results['pred_verts_tpose'][i]
            pve_tpose2 = pred_results['gt_verts_tpose'][i]
            single_data['pve_tpose'] = np.average(
                np.linalg.norm(pve_tpose1-pve_tpose2, axis=1))
            self.pred_results.append(single_data)


    def remove_redunc(self):
        new_pred_results = list()
        img_id_set = set()
        for data in self.pred_results:
            data_idx = data['data_idx']
            img_id = self.data_list[data_idx]['image_path']
            if img_id not in img_id_set:
                new_pred_results.append(data)
                img_id_set.add(img_id)
        self.pred_results = new_pred_results
        print("Number of test data:", len(self.pred_results))


    @property
    def pve(self):
        res = np.average([data['pve']
                          for data in self.pred_results])
        return res

    @property
    def mpjpe(self):
        res = np.average([data['mpjpe']
                          for data in self.pred_results])
        return res

    @property
    def pve_tpose(self):
        res = np.average([data['pve_tpose']
                          for data in self.pred_results])
        return res


    def build_dirs(self, res_dir):
        for i, result in enumerate(self.pred_results):
            img_path = self.data_list[result['data_idx']]['image_path']
            img = eval_utils.pad_and_resize(cv2.imread(img_path))
            subdir = eval_utils.get_subdir(img_path)
            res_subdir = osp.join(res_dir, subdir)
            if not osp.exists(res_subdir):
                os.makedirs(res_subdir)


    def visualize_result_single(self, start, end, res_dir, render_util):
        from smpl_webuser.serialization import load_model
        smpl_origin = load_model(osp.join(self.model_root, 'smpl_cocoplus_neutral.pkl'))

        for i, result in enumerate(self.pred_results[start:end]):
            # get result subdir path and image file path
            img_path = self.data_list[result['data_idx']]['image_path']
            img_name = img_path.split('/')[-1]
            subdir = eval_utils.get_subdir(img_path)
            res_subdir = osp.join(res_dir, subdir)
            res_img_path = osp.join(res_subdir, img_name)
            # render predicted smpl to image
            img = eval_utils.pad_and_resize(cv2.imread(img_path))
            render_img = render_util.render_smpl_to_image(
                img.copy(), result['vis_verts'], result['cam'], self.renderer)
            render_img = np.concatenate((img, render_img), axis=1)
            blank_img = np.ones((224,224,3),dtype=np.uint8)
            render_img = np.concatenate((render_img, blank_img), axis=1)
            # render predicted smpl in multi-view
            smpl = copy.deepcopy(smpl_origin)
            render_pred_smpl = render_util.render_image(smpl, result['smpl_pose'], result['smpl_shape'])
            # save image
            res_img = np.concatenate((render_img, render_pred_smpl), axis=0)
            cv2.imwrite(res_img_path, res_img)

            if i%10 == 0:
                print("{} Processed:{}/{}".format(os.getpid(), i, len(self.pred_results[start:end])))


    def visualize_result(self, res_dir):
        assert sys.version_info[0] == 2, "This code could only run in Python 2"
        from models.renderer import SMPLRenderer
        os.chdir('src')
        import util.render_util as render_util
        os.chdir('../')

        self.renderer = SMPLRenderer(
            img_size=224, face_path=osp.join(self.model_root,'smpl_faces.npy'))
        # build result subdirs first
        self.build_dirs(res_dir)
        # start processing
        num_process = 8
        num_each = len(self.pred_results) // num_process
        process_list = list()
        for i in range(num_process):
            start = i*num_each
            end = (i+1)*num_each if i<num_process-1 else len(self.pred_results)
            p = mp.Process(target=self.visualize_result_single, args=(start, end, res_dir, render_util))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()


    def save_to_pkl(self, res_pkl_file):
        saved_data = dict(
            model_root=self.model_root,
            data_list=self.data_list,
            pred_results=self.pred_results
        )
        pio.save_pkl_single(res_pkl_file, saved_data, protocol=2)

    def load_from_pkl(self, pkl_path):
        saved_data = pio.load_pkl_single(pkl_path)
        self.data_list = saved_data['data_list']
        self.pred_results = saved_data['pred_results']
        self.model_root=saved_data['model_root']


def main():
    os.chdir('../')
    evaluator = Evaluator()
    if len(sys.argv) > 1:
        epoch = sys.argv[1]
        pkl_path = 'evaluate_results/estimator_{}.pkl'.format(epoch)
        res_dir = 'evaluate_results/images/{}'.format(epoch)
    else:
        pkl_path = 'evaluate_results/estimator.pkl'
        res_dir = 'evaluate_results/images/default'
    ry_utils.renew_dir(res_dir)
    start = time.time()
    evaluator.load_from_pkl(pkl_path)
    end = time.time()
    print("Load evaluate results complete, time costed : {0:.3f}s".format(end-start))
    evaluator.visualize_result(res_dir)


if __name__ == '__main__':
    main()