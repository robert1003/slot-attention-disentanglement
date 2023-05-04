import argparse
import torch
from dataset import *
from model import *
from tqdm import tqdm
from metrics import adjusted_rand_index
from torchinfo import summary
import torchvision

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
            resolution = (128, 128)
            downsample_dim = 128
        elif opt.dinosaur_heavydownsample:
            resolution = (64, 64)
            downsample_dim = 64
        test_set = COCO2017Embeddings(data_path=opt.dataset_path, embed_path=opt.embed_path, 
                                       split='val', resolution=resolution, dynamic_load=opt.coco_mask_dynamic,
                                       downsample_mask=downsample_dim)
    elif opt.dataset == "clevr":
        val_set = CLEVR(path=opt.dataset_path, split="val",
                rescale=opt.dataset_rescale)
        mdsprites = False
    elif opt.dataset == 'multid-gray':
        assert "test" in opt.dataset_path
        assert opt.num_slots == 6, "Invalid number of slots for MultiDSpritesGrayBackground"
        test_set = MultiDSpritesGrayBackground(path=opt.dataset_path,
                rescale=opt.dataset_rescale)
        resolution = (64, 64)
        mdsprites = True
    else:
        assert "test" in opt.dataset_path
        assert opt.num_slots == 5, "Invalid number of slots for MultiDSpritesColorBackground"
        test_set = MultiDSpritesColorBackground(path=opt.dataset_path,
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

    ckpt = torch.load(opt.model_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    first = True
    first_sample = None
    first_predict = None
    cov_mean = 0
    ari = []
    for sample in tqdm(test_dataloader):
        image = sample['image'].to(device)
        if not opt.dinosaur:
            image = image.to(device)
        if opt.dinosaur_downsample:
            image = torch.nn.functional.interpolate(image, size=(128, 128))
        elif opt.dinosaur_heavydownsample:
            image = torch.nn.functional.interpolate(image, size=(64, 64))

        if first:
            first_sample = sample

        if first and opt.print_model:
            if opt.base:
                summary(model, input_data=image)
            elif opt.dinosaur:
                embedding = sample['embedding'].to(device)
                summary(model, input_data=(embedding, True))
            else:
                summary(model, input_data=(image, True))

        with torch.no_grad():
            if opt.base:
                recon_combined, recons, masks, slots = model(image)
            elif opt.dinosaur:
                embedding = sample['embedding'].to(device)
                embedding = sample['embedding'].to(device)
                recon_combined, recons, masks, slots, proj_loss_dict = model(embedding, True)
            else:
                recon_combined, recons, masks, slots, proj_loss_dict = model(image, False)

        feature_cov = torch.cov(slots.detach().reshape((-1,) + slots.shape[2:]).T).cpu()
        cov_mean += feature_cov.abs().mean()

        if opt.dataset_rescale:
            recon_combined = (recon_combined + 1.) / 2.
            recons = (recons + 1.) / 2.

        if first:
            first_predict = (recon_combined, recons, masks, slots)

        flattened_masks = torch.flatten(masks, start_dim=2, end_dim=4)
        flattened_masks = torch.permute(flattened_masks, (0, 2, 1))
        ari.append(adjusted_rand_index(sample['mask'].to(device), flattened_masks.to(device)))

        first = False

    ari = torch.concatenate(ari)
    print('NaN:', torch.isnan(ari).sum().item(), '/', ari.shape[0])
    ari = ari[~torch.isnan(ari)]

    print('ARI on {} samples: {}'.format(len(test_set), ari.mean().item()))
    print('slot feature mean', cov_mean / len(test_dataloader))

    if opt.visualize_mask:
        image = first_sample['image'].permute(0, 2, 3, 1).cpu().numpy()
        h, w = image.shape[1], image.shape[2]
        batch_size, hw, max_obj = first_sample['mask'].shape
        assert h*w == hw
        gt_mask = first_sample['mask'].reshape(batch_size, h, w, max_obj).bool().int() \
                .permute(0, 3, 1, 2).unsqueeze(4).repeat(1, 1, 1, 1, 3).cpu().numpy()
        pred_mask = first_predict[2].cpu()
        _, ids = torch.max(pred_mask, dim=1, keepdim=True)
        max_mask = torch.zeros_like(pred_mask)
        max_mask.scatter_(1, ids, 1)
        max_mask = max_mask.cpu().numpy()

        color_mask = [
            np.array([[[200, 100, 100]]]).repeat(h, axis=0).repeat(w, axis=1)   / 255., # red
            np.array([[[135, 135, 50]]]).repeat(h, axis=0).repeat(w, axis=1)   / 255., # green
            np.array([[[40, 80, 110]]]).repeat(h, axis=0).repeat(w, axis=1)   / 255., # blue
            np.array([[[165, 165, 80]]]).repeat(h, axis=0).repeat(w, axis=1) / 255., # yellow
            np.array([[[65, 130, 140]]]).repeat(h, axis=0).repeat(w, axis=1) / 255., # cyan
            np.array([[[100, 75, 60]]]).repeat(h, axis=0).repeat(w, axis=1) / 255., # brown
            np.array([[[190, 125, 65]]]).repeat(h, axis=0).repeat(w, axis=1) / 255., # orange
        ]
        assert opt.num_slots <= len(color_mask)

        fig, axs = plt.subplots(1, 7, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.05)
        for i in range(7):
            axs[i].imshow(image[i])
            tot_mask = 0.0
            color_idx = 0
            for j in range(max_obj):
                if gt_mask[i][j].sum() > 0:
                    tot_mask += color_mask[color_idx] * gt_mask[i][j]
                    color_idx += 1
                    if color_idx >= 7:
                        break
            axs[i].axis('off')
            axs[i].imshow(tot_mask, alpha=0.5)
        plt.savefig('visualize_gt_mask.png', bbox_inches='tight')
        plt.clf()

        fig, axs = plt.subplots(1, 7, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.05)
        for i in range(7):
            axs[i].imshow(image[i])
            tot_mask = np.sum([color_mask[j] * max_mask[i][j] for j in range(opt.num_slots)], axis=0)
            axs[i].axis('off')
            axs[i].imshow(tot_mask, alpha=0.7)
        plt.savefig('visualize_mask.png', bbox_inches='tight')
        plt.clf()

    if opt.visualize_recon:
        image = first_sample['image']
        recon_combined = first_predict[0]
        recons = first_predict[1]
        masks = first_predict[2]

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
        torchvision.utils.save_image(images_to_show, 'visualize_recon.png')
    

if __name__ == '__main__':
    # parameters from train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_model', action='store_true', help='print model structure')
    parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size of slot MLP')
    parser.add_argument('--decoder_init_size', default=8, type=int, help='(dinosaur+COCO only) init size for adaptive decoders')
    parser.add_argument('--decoder_hid_dim', default=64, type=int, help='(dinosaur+COCO only) hidden dimension size of decoder MLP')
    parser.add_argument('--decoder_num_conv_layers', default=6, type=int, help='(dinosaur+COCO only) number of conv layers in decoder')
    parser.add_argument('--decoder_type', choices=['adaptive', 'fixed', 'coco-adaptive', 'coco-fixed'], help='(dinosaur+COCO only) type of decoder to use')
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
    parser.add_argument('--info-nce', action='store_true', help='use InfoNCE style loss instead of cov loss')
    parser.add_argument('--info-nce-weight', default=0.1, type=float, help='weight given to the info nce loss')
    parser.add_argument('--info-nce-warmup', default=5000, type=float, help='number of warup steps for the infonce loss')
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

    # custom parameters for eval.py
    parser.add_argument('--visualize-mask', action='store_true', help='visualize the slot recon mask')
    parser.add_argument('--visualize-recon', action='store_true', help='visualize the slot recon image')

    main(parser.parse_args())
