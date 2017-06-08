import os
import csv
import cv2
import numpy as np
from torch.autograd import Variable

from datasets.flower_testset import get_loader
from tnn.network.tester import Tester
from tnn.utils.path import mkdir
from tnn.utils.meter import AverageValueMeter
from network.resnet import resnet50, accuracy

params = Tester.TestParams()

# hyper-parameters
# ====================================
data_dir = 'data/flower/val'
n_classes = len(os.listdir(data_dir))

inp_size = 256
n_threads = 6

params.exp_name = 'flower_resnet50_12'
params.save_dir = 'models/training/{}'.format(params.exp_name)
# params.ckpt = None
params.ckpt = os.path.join(params.save_dir, 'ckpt_27_0.1972.h5.best')

# Don't use gpu if `gpus == None`
params.gpus = [0]
params.batch_size = 16 * len(params.gpus)

params.print_freq = 20

output_dir = 'results'
# ====================================
mkdir(output_dir)

test_data = get_loader(data_dir, inp_size, params.batch_size,
                        shuffle=False, num_workers=n_threads)
print('test dataset len: {}'.format(len(test_data.dataset)))

# model
model = resnet50(num_classes=n_classes, inp_size=inp_size)
print('num_classes: {}'.format(n_classes))


def batch_processor(state, batch):
    gpus = state.params.gpus
    im, fname, target = batch
    volatile = not state.model.training
    im_var = Variable(im, volatile=volatile).cuda(device_id=gpus[0])

    inputs = [im_var]
    gts = None
    saved_for_eval = [fname, target]

    return inputs, gts, saved_for_eval

tester = Tester(model, params, batch_processor, test_data)

results = []
top1 = AverageValueMeter()
top3 = AverageValueMeter()
top5 = AverageValueMeter()
for i, (output, saved_for_eval) in enumerate(tester.test()):
    fnames, targets = saved_for_eval
    target_data = targets.cuda(output.data.get_device())
    prec1 = accuracy(output.data, target_data, topk=(1, 3, 5))
    top1.add(prec1[0][0])
    top3.add(prec1[1][0])
    top5.add(prec1[2][0])

    predicts = output.data.squeeze().cpu().numpy()
    for i, fname in enumerate(fnames):
        results.append((fname, targets[i], predicts[i]))

print('-----------------')
print('top1: {}, top3: {}, top5: {}'.format(top1.value(), top3.value(), top5.value()))
print(len(results))

result_file = os.path.join(output_dir, '{}.csv'.format(params.exp_name))

classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
classes.sort()
with open(result_file, 'w') as f:
    spamwriter = csv.writer(f, delimiter=',')
    titles = ['class', 'image_name']
    titles.extend(classes)
    spamwriter.writerow(titles)
    for fname, target, result in results:
        to_write = [target, fname] + result.tolist()
        spamwriter.writerow(to_write)
print('save to {}'.format(result_file))
