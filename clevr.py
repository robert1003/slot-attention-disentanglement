import torch
import torchvision
import glob
import os

class CLEVR(torch.utils.data.Dataset):
    """
        Loads CLEVR dataset images
        https://cs.stanford.edu/people/jcjohns/clevr/
    """

    def __init__(self, path, resolution, partition="train"):
        # self.opt = opt
        self.partition = partition
        self.all_files = glob.glob(os.path.join(path, partition, f'CLEVR_{partition}_*'))
        self.transforms = torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR))
        )   
        self.loader = torchvision.datasets.folder.default_loader
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
    

