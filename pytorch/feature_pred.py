import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, r2_score

import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

import torchvision

import wandb
import matplotlib 
import matplotlib.pyplot as plt


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Datasets --> TODO(as) have to adjust this to load images + features
    resolution = (64, 64)
    assert opt.num_slots == 5, "Invalid number of slots for MultiDSpritesColorBackground"
    train_set = MultiDSpritesColorFeatures(path=opt.dataset_path, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    
    val_set = MultiDSpritesColorFeatures(path=opt.dataset_path, split='val')
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    
    test_set = MultiDSpritesColorFeatures(path=opt.test_path, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers)

    # Model
    model = MDSpritesFeaturePrediction(opt, resolution).to(device)

    # Optimizer
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=opt.learning_rate)

    # Training
    wandb.init(project="vlr_slot_attn", entity="vlr-slot-attn", config=opt)
    start = time.time()
    run_id, run_name = wandb.run.id, wandb.run.name
    pbar = tqdm(total=opt.num_train_steps)
    i = 0
    total_loss = 0
    prev_val_loss = -1
    early_stop_cnt = 0
    pbar.update(i)
    while i < opt.num_train_steps and early_stop_cnt < 3:
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

            pred, recon_combined, recons, masks = model(sample['image'].to(device))
            loss, indices = match_loss_calc(pred, sample, device)

            if False:
                # Visualize image, reconstruction, and slots for subset of images
                images_to_show = []
                rec = recons[0].permute(0,3,1,2).cpu().detach()
                msk = masks[0].permute(0,3,1,2).cpu().detach()
                images_to_show.append(sample['image'].to(device)[0])
                images_to_show.append(recon_combined[0].cpu().detach())
                for j in range(opt.num_slots):
                    images_to_show.append(rec[j] * msk[j] + (1 - msk[j]))
                images_to_show = torchvision.utils.make_grid(images_to_show, nrow=opt.num_slots+2).clamp_(0,1)
                plt.imshow(images_to_show.permute((1, 2, 0)))

                for idx in range(sample['numerical'].shape[1]):
                    print(f"{idx}: {sample['numerical'][0][idx]}")

                print(" ")
                for idx in range(sample['numerical'].shape[1]):
                    print(f"{idx} ({int(sample['valid'][0][idx])}): {indices[0][0][idx]}, {indices[0][1][idx]}")

            vis_dict['loss'] = loss
            wandb.log(vis_dict, step=i)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.vis_freq > 0 and i % opt.vis_freq == 0:
                total_loss /= opt.vis_freq
                print ("Steps: {}, Loss: {}, Time: {}".format(i, total_loss,
                    datetime.timedelta(seconds=time.time() - start)))
                total_loss = 0

            if i % opt.early_stopping == 0:
                # pg 25, if in-distribution validation loss does not decrease by more than 0.01 for 3 evals (750 steps), training is stopped
                model.eval()
                val_loss = 0
                cnt = 0
                for sample in tqdm(val_dataloader):
                    pred, _, _, _ = model(sample['image'].to(device))
                    loss, _ = match_loss_calc(pred, sample, device)
                    val_loss += loss
                    cnt += 1
                val_loss /= cnt
                model.train()

                print ("Steps: {}, validation loss: {}".format(i, val_loss.item()))

                if prev_val_loss != -1 and prev_val_loss - val_loss < 0.01:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
                prev_val_loss = val_loss

            if i >= opt.num_train_steps:
                break
            pbar.update(1)

    # Save trained feature prediction model
    pbar.close()
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': i,
            'total_loss': total_loss,
            'wandb_run_id': run_id,
            }, opt.model_dir)

    # Evaluate on the test set
    eval(model, test_dataloader, device)

    wandb.finish()


def match_loss_calc(y_pred, sample, device):
    # Implement "loss matching" (https://arxiv.org/pdf/2107.00637.pdf pg3) since this is how primary paper results are presented
    y_numerical, y_categorical, y_valid = sample['numerical'].to(device), sample['categorical'].to(device), sample['valid']

    loss_mx = _loss_fn(y_pred, y_numerical, y_categorical)
    loss_per_object, indices = _make_matching(loss_mx, y_valid)

    def _get_ordered_objects(data, indices):
        # Use index broadcasting. The first indexing tensor has shape (B, 1),
        # `indices` has shape (B, min(num slots, num objects)).
        return data[torch.arange(data.shape[0]).unsqueeze(-1), indices]


    #y_num_matched = _get_ordered_objects(y_numerical, indices[:, 0])
    #y_cat_matched = _get_ordered_objects(y_categorical, indices[:, 0])
    #y_pred_matched = _get_ordered_objects(y_pred, indices[:, 1])

    # Calculate average per-object loss in the batch
    # TODO(as) check this normalization 
    return loss_per_object.sum() / (y_pred.shape[0] * y_pred.shape[1]), indices


def _loss_fn(y_pred, y_numerical, y_categorical):
    def cross_entropy(y, y_pred_logits):
        return (-(y * y_pred_logits.log_softmax(dim=-1))).sum(dim=-1)

    def mse(y, y_pred):
        return ((y - y_pred) ** 2).mean(dim=-1)

    numerical_criterion = mse
    categorical_criterion = cross_entropy

    # y: (batch, max_num_objects, n_features)
    # y_pred: (batch, n_slots, n_features)
    # loss_slots: (batch, max_num_objects, n_slots)
    loss_slots = torch.zeros(y_numerical.shape[0], y_numerical.shape[1], y_pred.shape[1], device=y_numerical.device)
    y_numerical = y_numerical.unsqueeze(2)
    y_categorical = y_categorical.unsqueeze(2)
    y_pred = y_pred.unsqueeze(1)

    # Evaluate numerical losses
    num_categories = 4
    for idx in range(y_pred.shape[2] - num_categories):
        loss_slots += numerical_criterion(y_numerical[:, :, :, idx].unsqueeze(-1), y_pred[:, :, :, idx].unsqueeze(-1))

    # Evaluate categorical losses
    loss_slots += categorical_criterion(y_categorical, y_pred[:, :, :, -num_categories:])
    return loss_slots



def _make_matching(loss_matrix, valid):
    # Background/non-visible/non-selected objects are always matched last (high cost).
    cost_matrix = loss_matrix.cpu().clone().detach() * valid + 100000 * (1 - valid)

    _, indices = _hungarian_algorithm(cost_matrix)

    # Select matches from the full loss matrix by broadcasting indices.
    # Output has shape (B, min(num slots, num objects)).
    batch_range = torch.arange(loss_matrix.shape[0]).unsqueeze(-1)
    loss_per_object = loss_matrix[batch_range, indices[:, 0], indices[:, 1]]
    return loss_per_object, indices


def _hungarian_algorithm(cost_matrix):
    """Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.

    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.

    The outputs are on the same device as `cost_matrix` but gradients are detached.

    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].

    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).

    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """

    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device), indices.to(device)




def eval(model, test_dataloader, device):
    print("Evaluating trained model on test set...")
    model.eval()
    if torch.backends.mps.is_available():
        device = 'mps'
    model = model.to(device)

    results = None
    num_targets = None
    cat_targets = None
    valid_targets = None
    for sample in tqdm(test_dataloader):
        pred, _, _, _ = model(sample['image'].to(device))

        # TODO(as) need to align objects and slots?

        if results is None:
            results = pred
            num_targets = sample['numerical']
            cat_targets = sample['categorical']
            valid_targets = sample['valid']
        else:
            results = torch.concat((results, pred))
            num_targets = torch.concat((num_targets, sample['numerical']))
            cat_targets = torch.concat((cat_targets, sample['categorical']))
            valid_targets = torch.concat((valid_targets, sample['valid']))


    # Mask out invalid objects
    results = results[valid_targets == 1].cpu().detach().numpy()
    num_targets = num_targets[valid_targets == 1].cpu().detach().numpy()
    cat_targets = cat_targets[valid_targets == 1].cpu().detach().numpy()

    # Numerical: # objects x (3 x color, 1 x scale, 1 x 'x', 1 x 'y')
    # Categorical: # objects x (one-hot shape (3 dims.)) --> TODO(as) this should only be 3 but there are 4 classes?
    test_r2 = {i:0 for i in ['color', 'scale', 'x', 'y']}
    test_acc = {}

    cat_pred = results[:, -4:].argmax(axis=1)
    cat_targets = cat_targets.argmax(axis=1)

    test_acc['shape'] = accuracy_score(cat_targets, cat_pred)
    test_r2['color'] = r2_score(num_targets[:, :3], results[:, :3])
    test_r2['scale'] = r2_score(num_targets[:, 3], results[:, 3])
    test_r2['x'] = r2_score(num_targets[:, 4], results[:, 4])
    test_r2['y'] = r2_score(num_targets[:, 5], results[:, 5])

    print(f"Test shape accuracy: {test_acc['shape']}")
    print(f"Test color R^2: {test_r2['color']}")
    print(f"Test scale R^2: {test_r2['scale']}")
    print(f"Test x R^2: {test_r2['x']}")
    print(f"Test y R^2: {test_r2['y']}")





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
    parser.add_argument('--dataset_path', default="./data/CLEVR_v1.0/images", type=str, help='path to train dataset')
    parser.add_argument('--test_path', default="./data/CLEVR_v1.0/images", type=str, help='path to test dataset')
    parser.add_argument('--base', action='store_true', help='run base Slot Attention model, no projection head')

    parser.add_argument('--vis_freq', default=50, type=int, help='frequency at which to generate visualization (in steps)')

    # NOTE: options are unused, only needed for compatibility with loaded checkpoint model (project head disabled during training of feature predictor)
    parser.add_argument('--proj_dim', default=1024, type=int, help='dimension of the projection space')
    parser.add_argument('--std_target', default=1.0, type=float, help='target std. deviation for each projection space dimension')
    parser.add_argument('--cov-div-sq', action='store_true', help='divide projection head covariance by the square of the number of projection dimensions')
    parser.add_argument('--slot-cov', action='store_true', help='calculate covariance over slots rather than over projection feature dimension')
    parser.add_argument('--bce-loss', action='store_true', help='calculate the reconstruction loss using binary cross entropy rather than mean squared error')
    parser.add_argument('--identity-proj', action='store_true', help='set projection to identity function. This option is equivalent to applying var/cov regularization on slot vectors directly')
    parser.add_argument('--proj-layernorm', action='store_true', help='use layernorm in projection layer (default is batchnorm)')

    main(parser.parse_args())



