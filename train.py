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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--name', required=True, type=str, help='name of the experiment' )
parser.add_argument('--model_dir', default='./checkpoint', type=str, help='where to save models' )
parser.add_argument('--log_dir', default='./logs', type=str, help='where to save logs' )
parser.add_argument('--image_dir', default='./images', type=str, help='where to save images' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

opt = parser.parse_args()

writer = SummaryWriter(os.path.join(opt.log_dir, opt.name))
os.makedirs(os.path.join(opt.model_dir, opt.name), exist_ok=False)
os.makedirs(os.path.join(opt.image_dir, opt.name), exist_ok=False)
resolution = (128, 128)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#train_set = CLEVR('train')
train_set = MultiDSprites('train')
model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

# dataset for visualization
test_set = MultiDSprites('test')
test_batch_size = 16
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=0)
test_images = next(iter(test_loader))['image']
test_images = test_images.to(device)

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

optimizer = optim.Adam(params, lr=opt.learning_rate)

start = time.time()
i = 0
for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0

    for sample in tqdm(train_dataloader):
        i += 1

        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (
            i / opt.decay_steps))

        optimizer.param_groups[0]['lr'] = learning_rate
        
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_dataloader)

    writer.add_scalar('Loss/train', total_loss, epoch)
    writer.add_scalar('Learning Rate', learning_rate, epoch)

    print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))

    if not epoch % 10:
        torch.save({
            'model_state_dict': model.state_dict(),
            }, os.path.join(opt.model_dir, opt.name, 'checkpoint-{}.ckpt'.format(epoch)))

        # visualize the reconstructions
        model.eval()
        recon_combineds, recons, masks, slots = model(test_images)

        images_to_show = []
        for i, image in enumerate(test_images):
            image = image.cpu()
            recon_combined = recon_combineds[i].cpu().detach()
            recon = recons[i].permute(0,3,1,2).cpu().detach()
            mask = masks[i].permute(0,3,1,2).cpu().detach()

            images_to_show.append(image)
            images_to_show.append(recon_combined)
            
            for j in range(opt.num_slots):
                picture = recon[j] * mask[j] + (1 - mask[j])
                images_to_show.append(picture)

        images_to_show = torchvision.utils.make_grid(images_to_show, nrow=opt.num_slots+2)
        print(images_to_show.shape)

        writer.add_image('Reconstructions', images_to_show, epoch)
        plt.figure(figsize=(15,2*test_batch_size))
        plt.imshow(images_to_show.permute(1,2,0))
        plt.savefig(os.path.join(opt.image_dir, opt.name, 'reconstructions-{}.png'.format(epoch)), bbox_inches='tight')
