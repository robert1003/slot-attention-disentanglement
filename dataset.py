import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PARTNET(Dataset):
    def __init__(self, split='train'):
        super(PARTNET, self).__init__()

        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = your_path     
        self.files = os.listdir(self.root_dir)
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, path, "0.png")).convert("RGB")
        image = image.resize((128 , 128))
        image = self.img_transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)

class CLEVR(Dataset):
    def __init__(self, path, split='train'):
        super(CLEVR, self).__init__()

        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = os.path.join(path, split)
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
    def __init__(self, path='./data/multi_dsprites/processed', split='train', unique=True):
        super(MultiDSprites, self).__init__()

        assert split in ['train', 'val', 'test']
        file_split = {'train': 'training', 'val': 'validation', 'test': 'test'}[split]
        self.split = split
        self.unique = unique
        self.data = np.load(os.path.join(path,
            file_split + '_images_rand4_' + ('unique' if unique else '') + '.npy'))
        self.mask = np.load(os.path.join(path,
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