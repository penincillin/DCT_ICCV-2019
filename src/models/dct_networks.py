
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from . import resnet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=True, num_classes=512)


class DCTEncoder(nn.Module):
    def __init__(self, opt):
        super(DCTEncoder, self).__init__()
        self.two_branch = opt.two_branch

        self.main_encoder = get_model(opt.main_encoder)
        if self.two_branch:
            self.aux_encoder = get_model(opt.aux_encoder)

        relu = nn.ReLU(inplace=False)
        if self.two_branch:
            fc2  = nn.Linear(2048, 1024)
        else:
            fc2  = nn.Linear(1024, 1024)

        classifier = nn.Linear(1024, opt.total_params_dim)
        feat_encoder = [relu, fc2, relu]
        regressor = [classifier, ]

        self.feat_encoder = nn.Sequential(*feat_encoder)
        self.regressor = nn.Sequential(*regressor)


    def forward(self, main_input, aux_input):
        main_feat = self.main_encoder(main_input)
        if self.two_branch:
            aux_feat = self.aux_encoder(aux_input)
            mid_feat = torch.cat([main_feat, aux_feat], dim=1)
        else:
            mid_feat = main_feat
        feat = self.feat_encoder(mid_feat)
        output = self.regressor(feat)
        return output
