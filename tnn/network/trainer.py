from __future__ import print_function

import os
import datetime
import numpy as np
from collections import OrderedDict
import shutil

import torch
import torch.nn as nn

from tnn.utils.timer import Timer
from tnn.utils.path import mkdir
import tnn.utils.meter as meter_utils
import tnn.network.net_utils as net_utils

try:
    import pycrayon
    from tnn.utils.crayon import CrayonClient
except ImportError:
    CrayonClient = None


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Trainer(object):

    class TrainParams(object):
        exp_name = 'Exp_name'
        init_lr = 0.01
        lr_decay_epoch = {5, 10, 20}
        lr_decay = 0.1

        batch_size = 32
        max_epoch = 30
        optimizer = None
        gpus = [0]
        save_dir = None
        ckpt = None
        re_init = True

        val_nbatch = 2
        val_nbatch_epoch = None

        save_freq = None

        print_freq = 20
        tensorboard_freq = 100
        tensorboard_hostname = None

    # hooks
    on_start_epoch_hooks = []
    on_end_epoch_hooks = []

    def __init__(self, model, train_params, batch_processor, train_data, val_data=None):
        assert isinstance(train_params, self.TrainParams)
        self.params = train_params
        self.train_data = train_data
        self.val_data = val_data
        self.val_stream = self.val_data.get_stream() if self.val_data is not None else None

        self.batch_processor = batch_processor

        self.batch_per_epoch = len(train_data)

        self.last_epoch = 0
        self.lr = self.params.init_lr
        self.optimizer = self.params.optimizer
        assert self.optimizer is not None, 'optimizer is not determined'

        self.log_values = OrderedDict()
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
        if ckpt is None and self.params.re_init:
            model.init_weight()
            print("re-init model weight")
        elif ckpt is not None:
            meta = net_utils.load_net(ckpt, model)
            if meta[0] >= 0:
                self.last_epoch = meta[0]
                self.lr = meta[1]
            print('load model from {}, last epoch: {}, lr: {}'.format(ckpt, self.last_epoch, self.lr))

        self.model = nn.DataParallel(model, device_ids=self.params.gpus)
        self.model = self.model.cuda(device_id=self.params.gpus[0])
        self.model.train()

        # tensorboard
        self.tf_exp = None
        if self.params.tensorboard_hostname is not None and CrayonClient is not None:
            cc = CrayonClient(hostname=self.params.tensorboard_hostname)
            exp_name = self.params.exp_name
            try:
                if self.last_epoch == 0:
                    cc.remove_experiment(exp_name)
                    exp = cc.create_experiment(exp_name)
                else:
                    exp = cc.open_experiment(exp_name)
            except ValueError:
                exp = cc.create_experiment(exp_name)
            self.tf_exp = exp

    def train(self):
        best_loss = np.inf
        for epoch in range(self.last_epoch, self.params.max_epoch):
            for fun in self.on_start_epoch_hooks:
                fun(self)

            self._train_epoch()

            for fun in self.on_end_epoch_hooks:
                fun(self)

            # save model
            save_name = 'ckpt_{}.h5'.format(self.last_epoch)
            save_to = os.path.join(self.params.save_dir, save_name)
            self._save_ckpt(save_to)

            # find best model
            if self.params.val_nbatch_epoch is not None:
                val_loss = self._val_epoch(self.params.val_nbatch_epoch)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_file = os.path.join(self.params.save_dir, 'ckpt_{}_{:.4f}.h5.best'.format(self.last_epoch, best_loss))
                    shutil.copyfile(save_to, best_file)
                    print('copy to {}'.format(best_file))

    def _save_ckpt(self, save_to):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        net_utils.save_net(save_to, model, epoch=self.last_epoch, lr=self.lr)
        print('save to {}'.format(save_to))

    def _process_log(self, src_dict, dest_dict):
        for k, v in src_dict.items():
            if isinstance(v, np.ndarray):
                # TODO: support image using visdom
                pass
            elif hasattr(v, '__iter__'):
                # a set of scale
                for i, x in enumerate(v):
                    sub_k = '{}_{}'.format(k, i)
                    dest_dict.setdefault(sub_k, meter_utils.AverageValueMeter())
                    dest_dict[sub_k].add(float(x))
            else:
                dest_dict.setdefault(k, meter_utils.AverageValueMeter())
                dest_dict[k].add(float(v))

    def _print_log(self, step, log_values, title=None):
        if title is None:
            print('epoch {}[{}/{}]'.format(self.last_epoch, step, self.batch_per_epoch), end='')
        else:
            print(title, end='')
        i = 0
        for k, v in log_values.items():
            if isinstance(v, meter_utils.AverageValueMeter):
                if i % 5 == 0 and i > 0:
                    print('\n\t\t', end='')
                else:
                    print(', ', end='')
                mean, std = v.value()
                print('{}: {:.4f}'.format(k, mean), end='')
                i += 1

        if title is None:
            # print time
            data_time = self.data_timer.duration
            batch_time = self.batch_timer.duration
            print(' ({:.2f}/{:.2f}s, fps:{:.1f}, rest: {})'.format(data_time, batch_time, self.params.batch_size/batch_time,
                                        str(datetime.timedelta(seconds=int((self.batch_per_epoch - step) * batch_time)))))
            self.batch_timer.clear()
            self.data_timer.clear()
        else:
            print()

    def _tensorboard_log(self, step, log_values, postfix=''):
        if self.tf_exp is None:
            return
        step = step + (self.last_epoch - 1) * self.batch_per_epoch
        for k, v in log_values.items():
            if isinstance(v, meter_utils.AverageValueMeter):
                self.tf_exp.add_scalar_value(k + postfix, v.value()[0], step=step)

    def _reset_log(self, log_values):
        for k, v in log_values.items():
            if isinstance(v, meter_utils.AverageValueMeter):
                v.reset()

    def _train_epoch(self):
        self.last_epoch += 1
        print('start epoch: {}'.format(self.last_epoch))

        if self.last_epoch in self.params.lr_decay_epoch:
            self.lr *= self.params.lr_decay
            adjust_learning_rate(self.optimizer, self.lr)
            print('adjust learning rate: {}'.format(self.lr))

        self.batch_timer.clear()
        self.data_timer.clear()
        self.batch_timer.tic()
        self.data_timer.tic()
        for step, batch in enumerate(self.train_data):
            inputs, gts, _ = self.batch_processor(self, batch)

            self.data_timer.toc()

            # forward
            output, saved_for_loss = self.model(*inputs)

            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)

            # save log
            self._process_log(saved_for_log, self.log_values)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.batch_timer.toc()

            # print log
            reset = False
            if step % self.params.print_freq == 0:
                self._print_log(step, self.log_values)
                reset = True

            if self.tf_exp is not None and step % self.params.tensorboard_freq == 0:
                self._tensorboard_log(step, self.log_values)
                reset = True

            # validation
            if step % self.params.tensorboard_freq == 0:
                val_logs = self._val_nbatch(self.params.val_nbatch)
                self._tensorboard_log(step, val_logs, postfix='_val')

            if self.params.save_freq is not None and step % self.params.save_freq == 0 and step > 0:
                save_to = os.path.join(self.params.save_dir,
                                       'ckpt_{}.h5.ckpt'.format((self.last_epoch - 1) * self.batch_per_epoch + step))
                self._save_ckpt(save_to)

            if reset:
                self._reset_log(self.log_values)

            self.data_timer.tic()
            self.batch_timer.tic()

    def _val_nbatch(self, n_batch):
        if self.val_stream is None:
            return

        self.model.eval()
        logs = OrderedDict()
        for i in range(self.params.val_nbatch):
            batch = next(self.val_stream)
            inputs, gts, _ = self.batch_processor(self, batch)

            output, saved_for_loss = self.model(*inputs)
            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)

            self._process_log(saved_for_log, logs)
        self._print_log(0, logs, title='Validation')

        self.model.train()
        return logs

    def _val_epoch(self, n_batch):
        self.model.eval()
        sum_loss = meter_utils.AverageValueMeter()
        print('val on validation set...')
        for step, batch in enumerate(self.val_data):
            if step > n_batch:
                break
            if step % self.params.print_freq == 0:
                print('[{}/{}]'.format(step, min(n_batch, len(self.val_data))))
            inputs, gts, _ = self.batch_processor(self, batch)
            output, saved_for_loss = self.model(*inputs)
            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)

            sum_loss.add(loss.data[0])

        mean, std = sum_loss.value()
        print('val: mean: {}, std: {}'.format(mean, std))
        self.model.train()
        return mean


