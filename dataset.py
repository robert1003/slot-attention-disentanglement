import os
import random
import json
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate




class CLEVR(Dataset):
    def __init__(self, split='train'):
        super(CLEVR, self).__init__()

        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = os.path.join('./data/CLEVR_v1.0/images', split)
        self.files = os.listdir(self.root_dir)
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
        image = image.resize((128 , 128))
        image = self.img_transform(image)
        sample = {'image': image}

        return sample

    def __len__(self):
        return len(self.files)
    



class MultiDSprites(Dataset):
    """
    Source: https://github.com/applied-ai-lab/genesis/blob/master/datasets/multid_config.py
    """
    def __init__(self, split='train', unique=True):
        super(MultiDSprites, self).__init__()

        assert split in ['train', 'val', 'test']
        file_split = {'train': 'training', 'val': 'validation', 'test': 'test'}[split]
        self.split = split
        self.unique = unique
        self.data = np.load(os.path.join('./data/multi_dsprites/processed',
            file_split + '_images_rand4_' + ('unique' if unique else '') + '.npy'))
        self.mask = np.load(os.path.join('./data/multi_dsprites/processed',
            file_split + '_masks_rand4_' + ('unique' if unique else '') + '.npy'))
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        image = (self.data[index]*255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize((128 , 128))
        image = self.img_transform(image)
        sample = {'image': image}

        return sample

    def __len__(self):
        return len(self.data)



    