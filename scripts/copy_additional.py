import os
import time
import shutil

types = ['Type_1', 'Type_2', 'Type_3']
root = 'data/ccs'
dst = 'data/ccs/train'

for t in types:
    src_root = os.path.join(root, t)
    for fname in os.listdir(src_root):
        name, ext = os.path.splitext(fname)
        dst_name = os.path.join(dst, t, fname)
        if os.path.isfile(dst_name):
            dst_name = os.path.join(dst, t, '{}_{}{}'.format(name, time.time(), ext))
        shutil.copyfile(os.path.join(src_root, fname), dst_name)
        print(dst_name)
