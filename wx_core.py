# -*- coding: UTF-8 -*-
from __future__ import print_function
import os
import urllib
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from network.resnet import resnet50
import torchvision.transforms as transforms
from datasets.flower_testset import im_loader
from tnn.utils.path import mkdir
from tnn.network import net_utils

from wechatpy import create_reply
import wechatpy.messages as messages
from wechatpy.replies import ArticlesReply
from wechatpy.utils import ObjectDict


# ============
site_root = 'http://166.111.139.148'
# ckpt = os.path.join('models/training/flower_resnet50', 'ckpt_24_0.2286.h5.best')
ckpt = os.path.join('models/training/flower_resnet50_12', 'ckpt_27_0.1972.h5.best')
inp_size = 256
gpu = 0

im_save_dir = '/data/flower/wx/saved'
classes_name_file = 'flower_names.txt'

# ============

class_names = []
im_urls = []
with open(classes_name_file, 'r') as f:
    for line in f.readlines():
        items = line.split(' ')
        if len(items) == 2:
            class_names.append(items[0].strip())
            im_urls.append(items[1].strip())
num_classes = len(class_names)
# print(class_names, im_urls)

im_transform = transforms.Compose([
            transforms.CenterCrop(inp_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# load model
model = resnet50(num_classes=num_classes)
net_utils.load_net(ckpt, model)
model = model.cuda(gpu)
model.eval()
print('load net from: {}'.format(ckpt))


def test_im(im_path):
    img = im_loader(im_path)
    img = im_transform(img)
    img_data = torch.unsqueeze(img, 0)

    im_var = Variable(img_data, volatile=True).cuda(gpu)
    output, _ = model(im_var)
    predicts = output.data.squeeze().cpu().numpy()
    return predicts


def msg_handler(msg):
    if msg.type == 'image':
        im_id = msg.media_id
        im_url = msg.image
        save_name = os.path.join(im_save_dir, '{}.jpg'.format(im_id))
        urllib.urlretrieve(im_url, save_name)
        # print('download image from: {} \nto: {}'.format(im_url, save_name))
        print('[log] recive an image...')

        # resize
        im = cv2.imread(save_name)
        if im is None:
            reply = create_reply('图片上传失败', msg)
        else:
            min_size = min(im.shape[:2])
            scale = 300. / min_size
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            cv2.imwrite(save_name, im)
            print('[log] resize image: {}'.format(scale))

            pred = test_im(save_name)
            inds = np.arange(len(pred), dtype=np.int)
            inds = sorted(inds, key=lambda i: pred[i], reverse=True)
            print('[log] predict: ', end='')
            print(inds)

            flowers = []
            for ind in inds[:5]:
                flower = {
                    'title': '{}: {:.3f}'.format(class_names[ind], pred[ind]),
                    'description': '{}: {:.3f}'.format(class_names[ind], pred[ind]),
                    'image': im_urls[ind],
                    'url': 'https://www.baidu.com/s?ie=UTF-8&wd={}'.format(urllib.quote(class_names[ind]))
                    # 'url': site_root + '/flower/{}.jpg'.format(ind),
                }
                flowers.append(flower)
            reply = create_reply(flowers, msg)
    else:
        text = '上传照片识别花名\n目前支持: '
        for name in class_names:
            text += '{} '.format(name)
        text += '{}种花'.format(len(class_names))
        reply = create_reply(text, msg)
        # reply = create_reply('Sorry, can not handle this for now', msg)

    return reply
