import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm
import timm
from timm.models.vision_transformer import VisionTransformer, checkpoint_seq

from dataset import COCO2017



class FrozenViTExperiment(VisionTransformer):
    def __init__(self):
        super().__init__()
        self.head = nn.Identity()       # make compatible with ViT state dict

    def forward(self, x):
        """Taken from timm/models/vision_transformer.py VisionTransformer.forward_features
        This experiment uses the final Transformer block output, without the final layer norm 
        and with the CLS token removed."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        # x = self.norm(x)

        # CLS token is prepended, DINOSAUR experiment removes it
        x = x[:, 1:, :]
        # assert x.shape[1] == 196
        return x


def generate_coco_embeddings(args):
    # Load pretrained weights (frozen ViT-B/16 encoder) and copy to a class with custom forward method
    timm_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
    model = FrozenViTExperiment()
    model.load_state_dict(timm_model.state_dict())
    model.eval()

    # Device management
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        # Mac M1 GPU support
        device =  torch.device('mps')
    model = model.to(device)

    # Load COCO 2017 dataset
    dataset = COCO2017(args.dataset_path, args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Generate embeddings and store them to output destination
    with torch.no_grad():
        for batch in tqdm(loader):
            out = model(batch['image'].to(device))

            # Store embeddings for each image to individual files in output directory
            # (Store embeddings separately so they can be matched with targets by filename)
            for embed, name in zip(out, batch['src']):
                out_name = os.path.join(args.output_dir, name.split('.')[0] + '.npy')
                np.save(out_name, embed.cpu().numpy())
                


if __name__ == '__main__':
    # Generate embeddings for COCO 2017 dataset from frozen ViT encoder for later use
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./data/coco_embeddings', type=str, help='where to save embeddings' )
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--dataset_path', default="./data/coco", type=str, help='path to COCO dataset')
    parser.add_argument('--split', default='train', type=str)

    generate_coco_embeddings(parser.parse_args())


