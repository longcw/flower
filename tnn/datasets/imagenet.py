import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataloader import sDataLoader

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def im_loader(path):
    return Image.open(path).convert('RGB')


def get_loader(data_dir, inp_size, batch_size, training=True, shuffle=True, num_workers=3):
    transform_train = transforms.Compose([
        transforms.RandomSizedCrop(inp_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform = transform_train if training else transform_test
    dataset = datasets.ImageFolder(data_dir, transform, loader=im_loader)

    data_loader = sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers)
    return data_loader
