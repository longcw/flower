import torch
import os
import tnn.network.net_utils as net_utils

from network.resnet import resnet50


pth_file = '../models/resnet50-19c8e357.pth'
h5_file = '../models/resnet50.h5'

model = resnet50()
model.load_state_dict(torch.load(pth_file))
print('load from: {}'.format(pth_file))

net_utils.save_net(h5_file, model)
print('save to: {}'.format(h5_file))