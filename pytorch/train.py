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

from metrics import adjusted_rand_index
import torchvision

import wandb
import matplotlib 
matplotlib.use('Agg')       # non-interactive backend for matplotlib
import matplotlib.pyplot as plt


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolution = (128, 128)

    if opt.dinosaur:
        # Resolution is hard-coded by the frozen ViT input dimensions
        resolution = (224, 224)
        downsample_dim = 0
        if opt.dinosaur_downsample:
            downsample_dim = 128
        elif opt.dinosaur_heavydownsample:
            downsample_dim = 64
        train_set = COCO2017Embeddings(data_path=opt.dataset_path, embed_path=opt.embed_path, 
                                       split='train', resolution=resolution, dynamic_load=opt.coco_mask_dynamic,
                                       downsample_mask=downsample_dim)
    elif opt.dataset == "clevr":
        train_set = CLEVR(path=opt.dataset_path, split="train",
                rescale=opt.dataset_rescale)
        mdsprites = False
    elif opt.dataset == 'multid-gray':
        assert opt.num_slots == 6, "Invalid number of slots for MultiDSpritesGrayBackground"
        train_set = MultiDSpritesGrayBackground(path=opt.dataset_path,
                rescale=opt.dataset_rescale)
        resolution = (64, 64)
        mdsprites = True
    else:
        assert opt.num_slots == 5, "Invalid number of slots for MultiDSpritesColorBackground"
        train_set = MultiDSpritesColorBackground(path=opt.dataset_path,
                rescale=opt.dataset_rescale)
        resolution = (64, 64)
        mdsprites = True

    if opt.base:
        model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim, sigmoid=opt.bce_loss, mdsprites=mdsprites).to(device)
    elif opt.dinosaur:
        # Hidden dimension must be dimension of ViT encoding for each token
        model = DINOSAURProjection(resolution, opt, vis=opt.vis_freq > 0).to(device)
    else:
        model = SlotAttentionProjection(resolution, opt, vis=opt.vis_freq > 0, mdsprites=mdsprites).to(device)

    if opt.bce_loss:
        # Assumes image is normalized to 0-1 range
        criterion = nn.BCELoss()
    else:
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
            image = sample['image']
            if not opt.dinosaur:
                image = image.to(device)
            if opt.dinosaur_downsample:
                image = torch.nn.functional.interpolate(image, size=(128, 128))
            elif opt.dinosaur_heavydownsample:
                image = torch.nn.functional.interpolate(image, size=(64, 64))
            vis_dict['learning_rate'] = learning_rate

            if i < opt.cov_warmup:
                cov_weight = opt.cov_weight * (i / opt.cov_warmup)
            else:
                cov_weight = opt.cov_weight
            
            if opt.base:
                recon_combined, recons, masks, slots = model(image)
                loss = criterion(recon_combined, image)
            else: 
                if opt.dinosaur:
                    embedding = sample['embedding'].to(device)
                    recon_combined, recons, masks, slots, proj_loss_dict = model(embedding, vis_step)
                else:
                    recon_combined, recons, masks, slots, proj_loss_dict = model(image, vis_step)
                proj_loss = opt.var_weight * proj_loss_dict["std_loss"] + cov_weight * proj_loss_dict["cov_loss"]
                if opt.dinosaur:
                    # Move reconstruction onto CPU for loss calculation, then back to GPU for backprop.
                    # Avoids every putting image on GPU, allowing for larger batch sizes
                    recon_loss = criterion(recon_combined.cpu(), image).to(device)
                else:
                    recon_loss = criterion(recon_combined, image)
                proj_loss *= opt.proj_weight
                loss = recon_loss + proj_loss
                vis_dict['recon_loss'] = recon_loss.item()
                vis_dict['std_loss'] = proj_loss_dict['std_loss'].item()
                vis_dict['cov_loss'] = proj_loss_dict['cov_loss'].item()

                # Visualize covariance loss with all weighting to make hyperparameter tuning easier
                vis_dict['weighted_cov_loss'] = opt.proj_weight * cov_weight * proj_loss_dict["cov_loss"]

                # Basic visualization of distribution of slot initializations in Slot Attn. module
                vis_dict['slot_sample_mean'] = torch.mean(model.slot_attention.slots_mu).item()
                vis_dict['slot_sample_std'] = torch.mean(model.slot_attention.slots_log_sigma).item()

                # Visualize the mean of the per-dimension means of slot projections 
                # (mean of per-slot means for covariance over slots setting)
                vis_dict['avg_proj_dim_mean'] = proj_loss_dict['proj_mean']

                # Visualize influence of reconstruction loss vs. projection head losses
                # Positive if reconstruction loss dominates loss, negative if projection losses dominate
                vis_dict['recon_proj_loss_diff'] = recon_loss - proj_loss

                # Visualize projection space batch size (sanity check that this is constant, 
                # otherwise may result in instabilities)
                vis_dict['proj_batch_sz'] = proj_loss_dict['proj_batch_sz']

            vis_dict['loss'] = loss
            if vis_step:
                if opt.dataset_rescale:
                    recon_combined = (recon_combined + 1.) / 2.
                    recons = (recons + 1.) / 2.

                if not opt.base:
                    vis_dict = visualize(vis_dict, opt, sample, image, recon_combined, recons, masks, slots, proj_loss_dict, train_set)
                else:
                    vis_dict = visualize(vis_dict, opt, sample, image, recon_combined, recons, masks, slots, None, train_set)
            wandb.log(vis_dict, step=i)
            
            total_loss += loss.item()
            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
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





def visualize(vis_dict, opt, sample, image, recon_combined, recons, masks, slots, proj_loss_dict, train_set):
    """Add visualizations to the dictionary to be logged with W&B"""
    if not opt.base:
        # Visualize projection space covariance matrix and feature std. dev. as heat maps
        # Resues proj_loss_dict from last step since model does not use projection head in eval
        plt.figure(figsize=(10,10))
        if opt.slot_cov:
            # When calculating slot covariance, generate a cov. matrix for each image in the batch.
            # Just sum over batch dimension here to get a matrix we can visualize
            plt.imshow(torch.sum(proj_loss_dict['cov_mx'], dim=0), cmap='Blues')
        else:
            plt.imshow(proj_loss_dict['cov_mx'], cmap='Blues')
        plt.colorbar()
        vis_dict['cov'] = wandb.Image(plt)
        plt.close()

        plt.figure(figsize=(10,10))
        if opt.slot_cov:
            plt.imshow(proj_loss_dict['std_vec'].flatten().unsqueeze(1).repeat((1, 50)), cmap='Blues')
        else:
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

    # Visualize ARI performance
    if 'mask' in sample or opt.coco_mask_dynamic:
        if 'mask' in sample:
            if opt.dinosaur:
                # Pad predicted masks to match number of GT masks
                num_pad = train_set.max_obj_per_image - masks.shape[1]
                if num_pad > 0:
                    padding = torch.zeros((masks.shape[0], num_pad, masks.shape[2], masks.shape[3], 1)).to(device)
                    masks = torch.concat((masks, padding), dim=1)

            flattened_masks = torch.flatten(masks, start_dim=2, end_dim=4)
            flattened_masks = torch.permute(flattened_masks, (0, 2, 1))
            vis_dict['ari'] = adjusted_rand_index(sample['mask'].to(device), flattened_masks.to(device)).mean().item()
        else:
            ari_sum = 0
            for idx in range(sample['id'].shape[0]):
                # Load ground truth mask for a single sample
                gt_mask = train_set.load_mask(sample['id'][idx].item())
                pred_mask = masks[idx]

                # Pad predicted masks to match number of GT masks
                num_pad = gt_mask.shape[1] - pred_mask.shape[0]
                if num_pad > 0:
                    padding = torch.zeros((num_pad, pred_mask.shape[1], pred_mask.shape[2], 1))
                    pred_mask = torch.concat((pred_mask, padding), dim=0)
                
                flattened_masks = torch.flatten(pred_mask, start_dim=1, end_dim=3)
                flattened_masks = torch.permute(flattened_masks, (1, 0))
                ari_sum += adjusted_rand_index(gt_mask.to(device).unsqueeze(0), flattened_masks.to(device).unsqueeze(0)).item()
            vis_dict['ari'] = ari_sum / sample['id'].shape[0]

    return vis_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
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
    parser.add_argument('--dataset', choices=['clevr', 'multid', 'multid-gray'], help='dataset to train on')
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
    parser.add_argument('--cov-warmup', default=0, type=int, help='number of warmup steps for the covariance loss')
    parser.add_argument('--bce-loss', action='store_true', help='calculate the reconstruction loss using binary cross entropy rather than mean squared error')
    parser.add_argument('--identity-proj', action='store_true', help='set projection to identity function. This option is equivalent to applying var/cov regularization on slot vectors directly')
    parser.add_argument('--dataset-rescale', action='store_true', help='by default image is rescaled from [0,255] to [0,1]. This option enables rescale from [0,255] to [-1,1]. This option the one used in original Slot Attention paper')
    parser.add_argument('--dinosaur', action='store_true', help='run projection head SA on top of frozen ViT embeddings')
    parser.add_argument('--embed_path', default="./data/coco/embedding", type=str, help='path to pre-generated COCO 2017 embeddings')
    parser.add_argument('--coco-mask-dynamic', action='store_true', help='load COCO masks only when they are needed for ARI calculation')
    parser.add_argument('--grad-clip', default=-1, type=float, help='Level of grad norm clipping to use. <0 to disable.')
    parser.add_argument('--proj-layernorm', action='store_true', help='use layernorm in projection layer (default is batchnorm)')
    parser.add_argument('--dinosaur-downsample', action='store_true', help='run DINOSAUR experiment with (128, 128) resolution rather than (224, 224)')
    parser.add_argument('--dinosaur-heavydownsample', action='store_true', help='run DINOSAUR experiment with (64, 64) resolution rather than (224, 224)')


    main(parser.parse_args())


