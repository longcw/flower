import os
import time
import shutil
import numpy as np

from tnn.utils.path import mkdir


src_root = '/data/flower/train'
dst_root = '/data/flower/val'
mkdir(dst_root)

types = os.listdir(src_root)
print(types)

val_rate = 0.05

for t in types:
    dst = os.path.join(dst_root, t)
    mkdir(dst)

    src = os.path.join(src_root, t)
    filenames = os.listdir(src)
    inds = np.arange(len(filenames))
    np.random.shuffle(inds)
    for i in range(int(len(filenames) * val_rate)):
        ind = inds[i]
        src_file = os.path.join(src, filenames[i])
        dst_file = os.path.join(dst, filenames[i])
        shutil.move(src_file, dst_file)

        print(dst_file)
