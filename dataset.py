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




class CLEVR(torch.utils.data.Dataset):
    """
        Loads CLEVR dataset images
        https://cs.stanford.edu/people/jcjohns/clevr/
    """

    def __init__(self, path, resolution, partition="train"):
        # self.opt = opt
        self.partition = partition
        self.all_files = glob.glob(os.path.join(path, partition, f'CLEVR_{partition}_*'))
        self.transforms = transforms.Compose(
            (transforms.ToTensor(),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR))
        )   
        self.loader = datasets.folder.default_loader
        self.resolution = resolution[0]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img = self.loader(self.all_files[idx])
        img = self.transforms(img)
        img = (img - 0.5) / 0.5
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        img = img.clamp(-1, 1)
        return img
    




class MultidSprites(Dataset):
    """
    Source: https://github.com/applied-ai-lab/genesis/blob/master/datasets/multid_config.py
    """
    def __init__(self, partition, resolution) -> None:
        super().__init__()

    def __len__():
        return 0
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    


    