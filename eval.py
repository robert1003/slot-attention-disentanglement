import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Image

from clevr import CLEVR
from dataset import *
from torch_model import *

# Hyperparameters.
def main():
    seed = 0
    batch_size = 1
    num_slots = 7
    num_iterations = 3
    resolution = (128, 128)
    device = "cpu"
    state_dict_path = '/Users/andrewstange/Desktop/CMU/Spring_2023/16-824/Project/torch_weight.pt'
    dataset_path = '/Users/andrewstange/datasets/CLEVR_v1.0/images/'

    model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, hid_dim=64)
    model.load_state_dict(torch.load(state_dict_path))

    dataset = CLEVR(dataset_path, resolution, 'train')
    
    model = model.to(device)
    image = dataset[4].to(device)

    renormalize = lambda x: x / 2. + 0.5 
    recon_combined, recons, masks, slots = model(image)
    recon_combined, recons, image = renormalize(recon_combined), renormalize(recons), renormalize(image)
    
    # Plotting
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
    image = image.squeeze(0)
    recon_combined = recon_combined.squeeze(0)
    recons = recons.squeeze(0)
    masks = masks.squeeze(0)
    image = image.permute(1,2,0).cpu().numpy()
    recon_combined = recon_combined.permute(1,2,0)
    recon_combined = recon_combined.cpu().detach().numpy()
    recons = recons.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')
    for i in range(7):
        picture = recons[i] * masks[i] + (1 - masks[i])
        ax[i + 2].imshow(picture)
        ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
        ax[i].grid(False)
        ax[i].axis('off')
    print("done")



if __name__ == "__main__":
    main()