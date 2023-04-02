import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import VisionDataset

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


class COCO2017(VisionDataset):
    """
    A dataset for generating ViT embeddings from the COCO 2017 dataset. Roughly
    based on torchvision.datasets.CocoDetection, with adaptations to give user
    access to filenames and pre-processed target masks. 

    Dataset source: 
    `wget http://images.cocodataset.org/zips/val2017.zip`
    `wget http://images.cocodataset.org/zips/train2017.zip`
    `wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

    Val size: 778 MB, 5k images
    Train size: 18 GB, ~118k images
    """
    def __init__(self, path='./data/coco', split='val', resolution=(224,224)):
        # Largely taken from torchvision.datasets.CocoDetection 
        assert split in ['train', 'val']
        root = os.path.join(path, f"{split}2017")
        annFile = os.path.join(path, f"annotations/instances_{split}2017.json")

        super().__init__(root, None, None, None)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.resolution = resolution
        
        # Resize to (224,224) ignoring aspect ratio (DINOSAUR pg35/36, COCO 2017 object segmentation)
        self.img_transform = transforms.Compose([transforms.Resize(size=resolution), transforms.ToTensor()])

    def __getitem__(self, index):
        id = self.ids[index]
        image, name = self._load_image(id)

        image = self.img_transform(image)
        sample = {'image': image, 'src' : name}
        return sample

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __len__(self):
        return len(self.ids)


class COCO2017Embeddings(COCO2017):
    """
    A dataset for handling pre-generated ViT embeddings of the COCO 2017 dataset.
    """
    def __init__(self, data_path='./data/coco', embed_path='./data/coco/embedding', split='val', resolution=(224, 224)):
        super().__init__(data_path, split, resolution)
        self.embed_path = embed_path

    def __getitem__(self, index):
        # TODO: convert target into per-pixel labels, and then resize target as well for evaluation
        id = self.ids[index]
        image, _ = self._load_image(id)
        #target = self._load_target(id)
        embed = self._load_embedding(id)

        image = self.img_transform(image)
        sample = {'image': image, 'embedding': embed} #, 'target' : target}
        return sample

    def _load_embedding(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"].split('.')[0] + '.npy'
        return torch.tensor(np.load(os.path.join(self.embed_path, path)))
    

