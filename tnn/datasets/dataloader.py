import os
from torch.utils.data.dataloader import DataLoader, DataLoaderIter


class sDataLoader(DataLoader):

    def get_stream(self):
        while True:
            for data in DataLoaderIter(self):
                yield data
