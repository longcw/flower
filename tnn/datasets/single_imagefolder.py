import torch.utils.data
import os
import cv2

from dataloader import sDataLoader


class SingleImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, ext='.jpg'):
        self.root = root

        filenames = [name for name in os.listdir(self.root) if os.path.splitext(name)[-1] == ext]
        self.filenames = sorted(filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        im_name = os.path.join(self.root, self.filenames[i])
        im = cv2.imread(im_name)

        return im


def collect_fn(data):
    return data[0] if len(data) == 1 else data


def get_loader(data_dir, ext='.jpg', batch_size=1, shuffle=False, num_workers=3):
    dataset = SingleImageFolder(data_dir, ext)

    data_loader = sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collect_fn)
    return data_loader
