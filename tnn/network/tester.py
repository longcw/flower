from __future__ import print_function

import os
import datetime
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from tnn.utils.timer import Timer
from tnn.utils.path import mkdir
import tnn.utils.meter as meter_utils
import tnn.network.net_utils as net_utils


class Tester(object):
    class TestParams(object):
        exp_name = 'Exp_name'

        batch_size = 32
        save_dir = None
        ckpt = None
        gpus = [0]

        print_freq = 20

    on_start_epoch_hooks = []

    def __init__(self, model, test_params, batch_processor, test_data):
        assert isinstance(test_params, self.TestParams)
        self.params = test_params
        self.test_data = test_data

        self.batch_processor = batch_processor
        self.batch_per_epoch = len(test_data)

        self.batch_timer = Timer()
        self.data_timer = Timer()

        mkdir(self.params.save_dir)
        # load model
        ckpt = self.params.ckpt
        if ckpt is None:
            ckpts = [fname for fname in os.listdir(self.params.save_dir) if os.path.splitext(fname)[-1] == '.h5']
            ckpt = os.path.join(
                self.params.save_dir, sorted(ckpts, key=lambda name: int(os.path.splitext(name)[0].split('_')[-1]))[-1]
            ) if len(ckpts) > 0 else None

        assert ckpt is not None

        meta = net_utils.load_net(ckpt, model)
        if meta[0] >= 0:
            self.last_epoch = meta[0]
            self.lr = meta[1]
        print('load model from {}, last epoch: {}, lr: {}'.format(ckpt, self.last_epoch, self.lr))

        self.model = nn.DataParallel(model, device_ids=self.params.gpus)
        self.model = self.model.cuda(device_id=self.params.gpus[0])
        self.model.eval()

    def test(self):

        for hook in self.on_start_epoch_hooks:
            hook(self)

        print('start eval...')

        self.data_timer.tic()
        self.batch_timer.tic()
        for step, batch in enumerate(self.test_data):
            inputs, _, saved_for_eval = self.batch_processor(self, batch)

            self.data_timer.toc()

            output, _ = self.model(*inputs)
            self.batch_timer.toc()

            if step % self.params.print_freq == 0:
                data_time = self.data_timer.duration
                batch_time = self.batch_timer.duration
                print('[{}/{}] ({:.2f}/{:.2f}s, fps:{:.1f}, rest: {})'.format(step, self.batch_per_epoch, data_time,
                                                                              batch_time,
                                                                              self.params.batch_size / batch_time,
                                                                              str(datetime.timedelta(
                                                                                  seconds=int((self.batch_per_epoch - step) * batch_time)))))

                self.batch_timer.clear()
                self.data_timer.clear()

            yield output, saved_for_eval

            self.batch_timer.tic()
            self.data_timer.tic()
