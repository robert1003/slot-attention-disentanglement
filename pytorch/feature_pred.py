















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

import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

import torchvision

import wandb
import matplotlib 
matplotlib.use('Agg')       # non-interactive backend for matplotlib
import matplotlib.pyplot as plt


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Datasets --> TODO(as) have to adjust this to load images + features
    resolution = (64, 64)
    assert opt.num_slots == 5, "Invalid number of slots for MultiDSpritesColorBackground"
    train_set = MultiDSpritesColorFeatures(path=opt.dataset_path)
        
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # Model
    model = MDSpritesFeaturePrediction(opt, resolution)

    # Optimizer + losses
    numerical_criterion = nn.MSELoss()
    categorical_criterion = nn.CrossEntropyLoss()
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=opt.learning_rate)

    # Training
    wandb.init(project="slot_attn", config=opt)#(project="vlr_slot_attn", entity="vlr-slot-attn", config=opt)
    start = time.time()
    run_id, run_name = wandb.run.id, wandb.run.name
    pbar = tqdm(total=opt.num_train_steps)
    i = 0
    pbar.update(i)
    while i < opt.num_train_steps:
        model.train()

        for sample in train_dataloader:
            i += 1
            vis_dict = {}
            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (i / opt.decay_steps))
            optimizer.param_groups[0]['lr'] = learning_rate
            vis_dict['learning_rate'] = learning_rate



            pred = model(sample['image'].to(device))


            # TODO(as) write matching + losses
            loss = numerical_criterion() + categorical_criterion()



            vis_dict['loss'] = loss
            wandb.log(vis_dict, step=i)
            total_loss += loss.item()
            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.vis_freq > 0 and i % opt.vis_freq == 0:
                total_loss /= opt.vis_freq
                print ("Steps: {}, Loss: {}, Time: {}".format(i, total_loss,
                    datetime.timedelta(seconds=time.time() - start)))
                total_loss = 0


            # TODO(as) implement early stopping (be careful with un-freezing base model)

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./tmp/feature10.ckpt', type=str, help='where to save models')
    parser.add_argument('--model_ckpt', default='./tmp/model10.ckpt', type=str, help='path to pretrained base model')
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')




    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=2000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_train_steps', default=6000, type=int, help='Number of training steps.')

    # pg 25, if in-distribution validation loss does not decrease by more than 0.01 for 3 evals (750 steps), training is stopped
    parser.add_argument('--early_stopping', default=250, type=int, help='number of steps to run evaluation for early stopping') 
    parser.add_argument('--dataset_path', default="./data/CLEVR_v1.0/images", type=str, help='path to dataset')
    parser.add_argument('--base', action='store_true', help='run base Slot Attention model, no projection head')

    parser.add_argument('--vis_freq', default=1000, type=int, help='frequency at which to generate visualization (in steps)')

    # NOTE: options are unused, only needed for compatibility with loaded checkpoint model (project head disabled during training of feature predictor)
    parser.add_argument('--proj_dim', default=1024, type=int, help='dimension of the projection space')
    parser.add_argument('--std_target', default=1.0, type=float, help='target std. deviation for each projection space dimension')
    parser.add_argument('--cov-div-sq', action='store_true', help='divide projection head covariance by the square of the number of projection dimensions')
    parser.add_argument('--slot-cov', action='store_true', help='calculate covariance over slots rather than over projection feature dimension')
    parser.add_argument('--bce-loss', action='store_true', help='calculate the reconstruction loss using binary cross entropy rather than mean squared error')
    parser.add_argument('--identity-proj', action='store_true', help='set projection to identity function. This option is equivalent to applying var/cov regularization on slot vectors directly')
    parser.add_argument('--proj-layernorm', action='store_true', help='use layernorm in projection layer (default is batchnorm)')

    main(parser.parse_args())



