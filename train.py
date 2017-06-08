import os
import numpy as np
import torch
from torch.autograd import Variable

from tnn.datasets.imagenet import get_loader
from network.resnet import resnet50
from tnn.network.trainer import Trainer

params = Trainer.TrainParams()

# hyper-parameters
# ====================================
data_dirs = 'data/flower'

inp_size = 256
n_threads = 10
momentum = 0.9
weight_decay = 5*1e-5

params.exp_name = 'flower_resnet50_12_256'
params.save_dir = 'models/training/{}'.format(params.exp_name)
params.ckpt = 'models/resnet50.h5'

params.max_epoch = 100
params.lr_decay_epoch = {20, 40, 60}
params.init_lr = 0.01
lr_decay = 0.1
# params.lr_decay = 0.1

params.gpus = [0]
params.batch_size = 32 * len(params.gpus)
params.val_nbatch = 2
params.val_nbatch_epoch = 100

params.print_freq = 20
params.tensorboard_freq = 100
# params.tensorboard_hostname = '166.111.139.98'
params.tensorboard_hostname = None
# ====================================


# load data
train_data = get_loader(os.path.join(data_dirs, 'train'), inp_size, params.batch_size, training=True,
                        shuffle=True, num_workers=n_threads)
print('train dataset len: {}'.format(len(train_data.dataset)))

valid_data = None
if params.val_nbatch > 0:
    valid_data = get_loader(os.path.join(data_dirs, 'val'), inp_size, params.batch_size, training=False,
                        shuffle=True, num_workers=6)
    print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
n_classes = len(os.listdir(os.path.join(data_dirs, 'train')))
print('num_classes: {}'.format(n_classes))
model = resnet50(num_classes=n_classes, inp_size=inp_size)

params.optimizer = torch.optim.SGD(model.parameters(), params.init_lr, momentum, weight_decay=weight_decay)


def batch_processor(state, batch):
    gpus = state.params.gpus
    im, targets = batch

    volatile = not state.model.training
    im_var = Variable(im, volatile=volatile).cuda(device_id=gpus[0])
    targets_var = Variable(targets).cuda(device_id=gpus[0])

    inputs = [im_var]
    gts = [targets_var]
    saved_for_eval = None

    return inputs, gts, saved_for_eval

trainer = Trainer(model, params, batch_processor, train_data, valid_data)

trainer.train()

