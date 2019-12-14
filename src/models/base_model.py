
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel():

    @property
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_info(self, save_info, epoch_label):
        save_filename = '{}_info.pth'.format(epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(save_info, save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, model_path):
        if self.opt.dist:
            network.module.load_state_dict(torch.load(
                model_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))
        else:
            saved_weights = torch.load(model_path)
            network.load_state_dict(saved_weights)

    def load_info(self, epoch_label):
        save_filename = '{}_info.pth'.format(epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        # saved_info = torch.load(save_path)
        if self.opt.dist:
            saved_info = torch.load(save_path, map_location=lambda storage, loc: storage.cuda(
                torch.cuda.current_device()))
        else:
            saved_info = torch.load(save_path)
        return saved_info

    def update_learning_rate(self):
        pass
