# import os
# import albumentations
# import numpy as np
import torch.nn as nn

# from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import torch
import torchvision

# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #


class DatasetH5(Dataset):
    def __init__(self, data, transform=None):
        super(DatasetH5, self).__init__()

        self.data = data
        self.transform = transform


    def __getitem__(self, index):
        x = self.data[index, ...]
        x = torch.tensor(x, dtype=torch.float32) / 127.5 - 1

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.data.shape[0]


def load_data(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.image_size, antialias=False),
            torchvision.transforms.CenterCrop(args.image_size),
        ]
    )
    file = h5py.File(args.dataset_path, "r")
    data = file["data"][()].copy()
    dataset = DatasetH5(data)
    print(len(dataset))
    print(args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader

# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
