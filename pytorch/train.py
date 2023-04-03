import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

import torchvision

import wandb
import matplotlib 
matplotlib.use('Agg')       # non-interactive backend for matplotlib
import matplotlib.pyplot as plt


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolution = (128, 128)

    if opt.dataset == "clevr":
        train_set = CLEVR(path=opt.dataset_path, split="train")
    else:
        train_set = MultiDSprites(path=opt.dataset_path, split='train')

    if opt.base:
        model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
    else:
        model = SlotAttentionProjection(resolution, opt, vis=opt.vis_freq > 0).to(device)

    criterion = nn.MSELoss()
    params = [{'params': model.parameters()}]
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    optimizer = optim.Adam(params, lr=opt.learning_rate)

    start = time.time()

    if os.path.isfile(opt.model_dir):
        print("Loading model from {}".format(opt.model_dir))
        ckpt = torch.load(opt.model_dir)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        i = ckpt['step']
        total_loss = ckpt['total_loss']
        wandb.init(project="vlr_slot_attn", entity="vlr-slot-attn", config=opt, id=ckpt['wandb_run_id'], resume="must")
    else:
        wandb.init(project="vlr_slot_attn", entity="vlr-slot-attn", config=opt)
        i = 0
        total_loss = 0

    run_id, run_name = wandb.run.id, wandb.run.name
    if opt.vis_freq > 0:
        # Log gradient every opt.vis_freq epoch
        wandb.watch(model, log='gradients', log_freq=opt.vis_freq)


    pbar = tqdm(total=opt.num_train_steps)
    pbar.update(i)
    while i < opt.num_train_steps:
        model.train()

        dataloader_len = len(train_dataloader)
        for sample in train_dataloader:
            i += 1
            vis_step = opt.vis_freq > 0 and i % opt.vis_freq == 0
            vis_dict = {}
            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (i / opt.decay_steps))
            optimizer.param_groups[0]['lr'] = learning_rate
            image = sample['image'].to(device)
            vis_dict['learning_rate'] = learning_rate
            
            if opt.base:
                recon_combined, recons, masks, slots = model(image)
                loss = criterion(recon_combined, image)
            else: 
                recon_combined, recons, masks, slots, proj_loss_dict = model(image, vis_step)
                proj_loss = opt.var_weight * proj_loss_dict["std_loss"] + opt.cov_weight * proj_loss_dict["cov_loss"]
                recon_loss = criterion(recon_combined, image)
                proj_loss *= opt.proj_weight
                loss = recon_loss + proj_loss
                vis_dict['recon_loss'] = recon_loss.item()
                vis_dict['std_loss'] = proj_loss_dict['std_loss'].item()
                vis_dict['cov_loss'] = proj_loss_dict['cov_loss'].item()

                # Basic visualization of distribution of slot initializations in Slot Attn. module
                vis_dict['slot_sample_mean'] = torch.mean(model.slot_attention.slots_mu).item()
                vis_dict['slot_sample_std'] = torch.mean(model.slot_attention.slots_sigma).item()

                # Visualize the mean of the per-dimension means of slot projections
                vis_dict['avg_proj_dim_mean'] = proj_loss_dict['proj_mean']

                # Visualize influence of reconstruction loss vs. projection head losses
                # Positive if reconstruction loss dominates loss, negative if projection losses dominate
                vis_dict['recon_proj_loss_diff'] = recon_loss - proj_loss

                # Visualize projection space batch size (sanity check that this is constant, 
                # otherwise may result in instabilities)
                vis_dict['proj_batch_sz'] = proj_loss_dict['proj_batch_sz']

            vis_dict['loss'] = loss
            if vis_step:
                if not opt.base:
                    vis_dict = visualize(vis_dict, opt, image, recon_combined, recons, masks, slots, proj_loss_dict)
                else:
                    vis_dict = visualize(vis_dict, opt, image, recon_combined, recons, masks, slots, None)
            wandb.log(vis_dict, step=i)
            
            total_loss += loss.item()
            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if vis_step:
                total_loss /= opt.vis_freq

                print ("Steps: {}, Loss: {}, Time: {}".format(i, total_loss,
                    datetime.timedelta(seconds=time.time() - start)))

                total_loss = 0

            if i % opt.store_freq == 0:
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': i,
                'total_loss': total_loss,
                'wandb_run_id': run_id,
                }, opt.model_dir)

            if i >= opt.num_train_steps:
                break
            pbar.update(1)

    pbar.close()
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': i,
            'total_loss': total_loss,
            'wandb_run_id': run_id,
            }, opt.model_dir)
    wandb.finish()





def visualize(vis_dict, opt, image, recon_combined, recons, masks, slots, proj_loss_dict):
    """Add visualizations to the dictionary to be logged with W&B"""
    if not opt.base:
        # Visualize projection space covariance matrix and feature std. dev. as heat maps
        # Resues proj_loss_dict from last step since model does not use projection head in eval
        plt.figure(figsize=(10,10))
        plt.imshow(proj_loss_dict['cov_mx'], cmap='Blues')
        plt.colorbar()
        vis_dict['cov'] = wandb.Image(plt)
        plt.close()

        plt.figure(figsize=(10,10))
        plt.imshow(proj_loss_dict['std_vec'].unsqueeze(1).repeat((1, 50)), cmap='Blues')
        plt.colorbar()
        vis_dict['std'] = wandb.Image(plt)
        plt.close()

    # Visualize image, reconstruction, and slots for subset of images
    images_to_show = []
    for i, img in enumerate(image[:16]):
        img = img.cpu()
        rec = recons[i].permute(0,3,1,2).cpu().detach()
        msk = masks[i].permute(0,3,1,2).cpu().detach()

        images_to_show.append(img)
        images_to_show.append(recon_combined[i].cpu().detach())
        
        for j in range(opt.num_slots):
            images_to_show.append(rec[j] * msk[j] + (1 - msk[j]))

    images_to_show = torchvision.utils.make_grid(images_to_show, nrow=opt.num_slots+2).clamp_(0,1)
    vis_dict['slot_output'] = wandb.Image(images_to_show)

    # Visualize slot feature covariance
    feature_cov = torch.cov(slots.detach().reshape((-1,) + slots.shape[2:]).T).cpu()
    plt.figure(figsize=(10,10))
    plt.imshow(feature_cov, cmap='Blues')
    plt.colorbar()
    vis_dict['slot_feature_cov'] = wandb.Image(plt)
    plt.close()

    if not opt.base:
        # Visualize projection head norms, calculated in ProjectionHead.forward
        vis_dict['proj_diff_norm'] = proj_loss_dict['proj_diff_norm']
        vis_dict['proj_out_norm'] = proj_loss_dict['proj_out_norm']
        vis_dict['proj_input_norm'] = proj_loss_dict['proj_input_norm']

    return vis_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0004, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_train_steps', default=500000, type=int, help='Number of training steps.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--dataset', choices=['clevr', 'multid'], help='dataset to train on')
    parser.add_argument('--dataset_path', default="./data/CLEVR_v1.0/images", type=str, help='path to dataset')
    parser.add_argument('--proj_dim', default=1024, type=int, help='dimension of the projection space')
    parser.add_argument('--proj_weight', default=1.0, type=float, help='weight given to sum of projection head losses')
    parser.add_argument('--var_weight', default=1, type=float, help='weight given to the variance loss')
    parser.add_argument('--cov_weight', default=25, type=float, help='weight given to the covariance loss')
    parser.add_argument('--base', action='store_true', help='run base Slot Attention model, no projection head')
    parser.add_argument('--vis_freq', default=1000, type=int, help='frequency at which to generate visualization (in steps)')
    parser.add_argument('--store_freq', default=10000, type=int, help='frequency at which to save model (in steps)')
    parser.add_argument('--std_target', default=1.0, type=float, help='target std. deviation for each projection space dimension')
    parser.add_argument('--cov-div-sq', action='store_true', help='divide projection head covariance by the square of the number of projection dimensions')
    parser.add_argument('--slot-cov', action='store_true', help='calculate covariance over slots rather than over projection feature dimension')

    main(parser.parse_args())


