import argparse
import torch
from dataset import *
from model import *
from tqdm import tqdm
from metrics import adjusted_rand_index

def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolution = (128, 128)

    if opt.dataset == "clevr":
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
        train_set = MultiDSprites(path=opt.dataset_path, split='train', num_slots=opt.num_slots,
                rescale=opt.dataset_rescale)
        mdsprites = True

    if opt.base:
        model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim, sigmoid=opt.bce_loss, mdsprites=mdsprites).to(device)
    else:
        model = SlotAttentionProjection(resolution, opt, vis=opt.vis_freq > 0, mdsprites=mdsprites).to(device)

    test_set = torch.utils.data.Subset(train_set, torch.arange(0, 320))
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    ckpt = torch.load(opt.model_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ari = []
    for sample  in test_dataloader:
        image = sample['image'].to(device)
        if opt.base:
            recon_combined, recons, masks, slots = model(image)
        else: 
            recon_combined, recons, masks, slots, proj_loss_dict = model(image, vis_step)

        if opt.dataset_rescale:
            recon_combined = (recon_combined + 1.) / 2.
            recons = (recons + 1.) / 2.

        flattened_masks = torch.flatten(masks, start_dim=2, end_dim=4)
        flattened_masks = torch.permute(flattened_masks, (0, 2, 1))
        ari.append(adjusted_rand_index(sample['mask'].to(device), flattened_masks.to(device)))

    ari = torch.concatenate(ari)
    print('NaN:', torch.isnan(ari).sum().item(), '/', ari.shape[0])
    ari = ari[~torch.isnan(ari)]

    print('ARI on first 320 of train dataset', ari.mean().item())

if __name__ == '__main__':
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

    main(parser.parse_args())
