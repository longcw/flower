import os
import cv2


root_dir = '/data/flower/train'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
for root, dirs, files in os.walk(root_dir):
    for fname in files:
        name, ext = os.path.splitext(fname)
        if ext in IMG_EXTENSIONS:
            fname = os.path.join(root, fname)
            im = cv2.imread(fname)
            if im is None:
                os.remove(fname)
                print(fname)
                continue
            min_size = min(im.shape[:2])
            scale = 1.
            if min_size > 800.:
                scale = 800. / min_size
                im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            cv2.imwrite(fname, im)
            print(fname, scale)

