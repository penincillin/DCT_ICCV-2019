
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--infer_dataset_path', type=str, default='', help='path of file or files that store the information of the test dataset')
        self.parser.add_argument('--infer_res_dir', type=str, default='inference_results', help='path of directory to store the results of evaluation_results')
        self.parser.add_argument('--visualize_eval', action='store_true')
        self.parser.add_argument('--trained_model_path', type=str, default='', help='path of trained model weights')
        self.isTrain = False
