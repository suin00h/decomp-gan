from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from dataset.dataset import get_loader_prebuilt
from model.gan import DecompGen, BaseGen, Disc

# hyperparameters
rank_size = 10
noise_size = 100
num_class = 10

epoch = 10
batch_size = 128
lr = 2e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset
train_loader = get_loader_prebuilt(batch_size)

# models, optimizers
decompose_gen = DecompGen(rank_size, noise_size, num_class)
base_gen = BaseGen(noise_size, num_class)
disc = Disc(num_class)

decompose_gen = decompose_gen.to(device)
disc = disc.to(device)

opt_decompose_gen = optim.Adam(decompose_gen.parameters())
opt_base_gen = optim.Adam(base_gen.parameters())
opt_disc = optim.Adam(disc.parameters())

crit = nn.BCELoss()

# train
for e in range(epoch):
    for images, labels in tqdm(train_loader):
        cur_batch_size = images.size(0)
        
        images = images.to(device)
        labels = labels.to(device)
        
        label_real = torch.full((cur_batch_size,), 1.0, device=device)
        label_fake = torch.full((cur_batch_size,), 0.0, device=device)
        
        # train generator
        decompose_gen.zero_grad()
        
        noise = torch.randn((cur_batch_size, noise_size), device=device)
        gen_label = torch.randint(0, num_class, (cur_batch_size,), device=device)
        
        gen_output = decompose_gen(noise, gen_label)
        
        disc_output = disc(gen_output, gen_label)
        
        loss_gen = crit(disc_output, label_real)
        loss_gen.backward()
        opt_decompose_gen.step()
        
        # train discriminator
        disc.zero_grad()
        
        disc_output = disc(images, labels)
        loss_disc = crit(disc_output, label_real)
        
        disc_output = disc(gen_output.detach(), gen_label)
        loss_disc += crit(disc_output, label_fake)
        
        loss_disc.backward()
        opt_disc.step()
