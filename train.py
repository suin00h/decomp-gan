import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch import optim

from dataset.dataset import get_loader_prebuilt, get_class_name
from model.gan import DecompGen, BaseGen, Disc

device = "cuda" if torch.cuda.is_available() else "cpu"
num_class = 10

# logging
logging_noise = torch.randn((16, 100), device=device)
logging_label = torch.randint(0, num_class, (16,), device=device)

class Logger():
    def __init__(
        self,
        logging_path: str
    ):
        self.logging_path = logging_path
        
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path)
    
    def log_msg(self, str):
        print('LOGGER: ' + str)
    
    def log_state(
        self,
        epoch: int,
        model_state,
        opt_state,
        loss_dict,
    ):
        torch.save({
            'epoch': epoch,
            'model': model_state,
            'optim': opt_state,
            'loss': loss_dict
        }, self.logging_path + f'{epoch}.pt')
    
    def log_image(
        self,
        epoch: int,
        images: Tensor,
    ):
        grid = torchvision.utils.make_grid(images, 4, normalize=True)
        grid = grid / 2 + 0.5
        grid = grid.numpy()
        img = np.transpose(grid, (1, 2, 0))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.savefig(self.logging_path + f'{epoch}.png')

logger = Logger(
    logging_path='./log/base_exp1/'
)

# hyperparameters
rank_size = 500
noise_size = 100

epoch = 200
batch_size = 128
lr = 2e-4

logger.log_msg('Hyperparameter Settings')
logger.log_msg(f'rank_size: {rank_size}')
logger.log_msg(f'noise_size: {noise_size}')
logger.log_msg(f'epoch: {epoch}')
logger.log_msg(f'batch_size: {batch_size}')
logger.log_msg(f'learning rate: {lr}')

# dataset
logger.log_msg(f'Loading CIFAR-10 Dataset')
train_loader = get_loader_prebuilt(batch_size)

# models, optimizers
decompose_gen = DecompGen(rank_size, noise_size, num_class)
base_gen = BaseGen(noise_size, num_class)
disc = Disc(num_class)

decompose_gen = decompose_gen.to(device)
base_gen = base_gen.to(device)
disc = disc.to(device)
disc.train()

opt_decompose_gen = optim.Adam(decompose_gen.parameters())
opt_base_gen = optim.Adam(base_gen.parameters())
opt_disc = optim.Adam(disc.parameters())

crit = nn.BCELoss()

# eval
if True:
    import sys
    from torchvision.utils import save_image
    
    # ckpt = torch.load('./log/deco_exp1/1999.pt')
    # decompose_gen.load_state_dict(ckpt['model'])
    # decompose_gen.eval()
    ckpt = torch.load('./log/base_exp1/199.pt')
    base_gen.load_state_dict(ckpt['model'])
    base_gen.eval()
    
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        cur_batch_size = images.size(0)
        noise = torch.randn((cur_batch_size, noise_size), device=device)
        # fake_images = decompose_gen(noise, labels)
        fake_images = base_gen(noise, labels)
        fake_images = (fake_images + torch.abs(torch.min(fake_images))) / (torch.max(fake_images) - torch.min(fake_images))
        images = images / 2 + 0.5
        
        for j in range(cur_batch_size):
            # save_image(images[j], f'./log/deco_exp1/real/{i * cur_batch_size + j}.png')
            # save_image(fake_images[j], f'./log/deco_exp1/fake/{i * cur_batch_size + j}.png')
            save_image(images[j], f'./log/base_exp1/real/{i * cur_batch_size + j}.png')
            save_image(fake_images[j], f'./log/base_exp1/fake/{i * cur_batch_size + j}.png')
    
    sys.exit()

# train
G_loss_list = []
D_loss_list = []

logger.log_msg(f'Predefined labels:')
logger.log_msg(f'{list(map(get_class_name, logging_label.cpu()))}')
logger.log_msg(f'Training ...')
for e in range(epoch):
    logger.log_msg(f'Epoch {e}')
    # decompose_gen = decompose_gen.train()
    base_gen = base_gen.train()
    
    batch_G_loss = []
    batch_D_loss = []
    for images, labels in tqdm(train_loader):
        cur_batch_size = images.size(0)
        
        images = images.to(device)
        labels = labels.to(device)
        
        label_real = torch.full((cur_batch_size,), 1.0, device=device)
        label_fake = torch.full((cur_batch_size,), 0.0, device=device)
        
        # train generator
        # decompose_gen.zero_grad()
        base_gen.zero_grad()
        
        noise = torch.randn((cur_batch_size, noise_size), device=device)
        gen_label = torch.randint(0, num_class, (cur_batch_size,), device=device)
        
        # gen_output = decompose_gen(noise, gen_label)
        gen_output = base_gen(noise, gen_label)
        
        disc_output = disc(gen_output, gen_label)
        
        loss_gen = crit(disc_output, label_real)
        loss_gen.backward()
        # opt_decompose_gen.step()
        opt_base_gen.step()
        
        batch_G_loss.append(loss_gen.item())
        
        # train discriminator
        disc.zero_grad()
        
        disc_output = disc(images, labels)
        loss_disc = crit(disc_output, label_real)
        
        disc_output = disc(gen_output.detach(), gen_label)
        loss_disc += crit(disc_output, label_fake)
        
        loss_disc.backward()
        opt_disc.step()
        
        batch_D_loss.append(loss_disc.item() / 2)
    
    G_loss_list.append(np.mean(batch_G_loss))
    D_loss_list.append(np.mean(batch_D_loss))
    
    if e % 10 == 0 or e == epoch-1:
        logger.log_msg(f'Logging checkpoint ... | G_loss: {G_loss_list[-1]:.2f} | D_loss: {D_loss_list[-1]:.2f}')
        # decompose_gen.eval()
        base_gen.eval()
        
        # fake_image = decompose_gen(logging_noise, logging_label)
        fake_image = base_gen(logging_noise, logging_label)
        logger.log_state(
            epoch=e,
            # model_state=decompose_gen.state_dict(),
            model_state=base_gen.state_dict(),
            # opt_state=opt_decompose_gen.state_dict(),
            opt_state=opt_base_gen.state_dict(),
            loss_dict={'G_loss': G_loss_list[-1], 'D_loss': D_loss_list[-1]}
        )
        logger.log_image(
            epoch=e,
            images=fake_image.detach().cpu()
        )

plt.figure(figsize=(10, 5))
plt.plot(G_loss_list, label="G")
plt.plot(D_loss_list, label="D")
plt.xlabel("epoch")
plt.ylabel("BCEloss")
plt.legend()
plt.savefig(logger.logging_path + 'loss.png')
