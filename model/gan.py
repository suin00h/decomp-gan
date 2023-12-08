import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

class DecompGen(nn.Module):
    """
    A conditional generator model with CP tensor decomposition method applied.
    
    Args:
        rank: number of ranks
        noise_size: size of input noise
        num_class: number of classes
    """
    def __init__(
        self,
        rank: int,
        noise_size: int,
        num_class: int
    ):
        super().__init__()
        
        self.noise_layer = nn.Sequential(
            nn.Linear(noise_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.label_layer = nn.Embedding(num_class, 128)
        
        # (B, 256, 1) -> (B, rank, 3)
        self.noise_to_channel = nn.Sequential(
            nn.ConvTranspose1d(256, rank, 3),
            nn.BatchNorm1d(rank),
            nn.LeakyReLU(0.2)
        )
        # (B, 256, 1) -> (B, rank, 32)
        self.noise_to_height = nn.Sequential(
            nn.ConvTranspose1d(256, rank // 4, 16),
            nn.BatchNorm1d(rank // 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(rank // 4, rank // 2, 16),
            nn.BatchNorm1d(rank // 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(rank // 2, rank, 2),
            nn.Tanh()
        )
        self.noise_to_width = nn.Sequential(
            nn.ConvTranspose1d(256, rank // 4, 16),
            nn.BatchNorm1d(rank // 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(rank // 4, rank // 2, 16),
            nn.BatchNorm1d(rank // 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(rank // 2, rank, 2),
            nn.Tanh()
        )
        
        self.coefficient = nn.Parameter(torch.rand(rank))
    
    def forward(
        self,
        noise: Tensor,
        label: Tensor
    ):
        """
        Args:
            noise: (B, noise_size) noise tensor
            label: (B) labels
        Returns:
            image: (B, 3, H, W) generated image
        """
        noise = self.noise_layer(noise)
        label = self.label_layer(label)
        latent = torch.cat((noise, label), dim=1)
        latent = latent.unsqueeze(-1)
        
        c_vectors = self.noise_to_channel(latent)
        h_vectors = self.noise_to_height(latent)
        w_vectors = self.noise_to_width(latent)
        
        recon_tensors = torch.einsum('brc,brh,brw->brchw', c_vectors, h_vectors, w_vectors)
        output = torch.einsum('brchw,r->bchw', recon_tensors, self.coefficient)
                
        return output

class BaseGen(nn.Module):
    def __init__(
        self,
        noise_size: int,
        num_class: int
    ):
        super().__init__()
        
        self.noise_layer = nn.Sequential(
            nn.Linear(noise_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.label_layer = nn.Embedding(num_class, 128)
        
        # (B, 256, 1) -> (B, 3, 32, 32)
        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(
        self,
        noise: Tensor,
        label: Tensor
    ):
        noise = self.noise_layer(noise)
        label = self.label_layer(label)
        latent = torch.cat((noise, label), dim=1)
        latent = latent.view(latent.size(0), latent.size(1), 1, 1)
        
        output = self.deconv_layer(latent)
        
        return output

class Disc(nn.Module):
    def __init__(
        self,
        num_class: int
    ):
        super().__init__()
        
        self.num_class = num_class
        # (B, 3, 32, 32) -> (B, 64, 16, 16)
        self.image_layer = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.label_embed = nn.Embedding(num_class, num_class * 32 * 32)
        # (B, 10, 32, 32) -> (B, 64, 16, 16)
        self.label_layer = nn.Sequential(
            nn.Conv2d(num_class, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        image: Tensor,
        label: Tensor
    ):
        image = self.image_layer(image)
        label = self.label_embed(label)
        label = label.view(-1, self.num_class, 32, 32)
        label = self.label_layer(label)
        latent = torch.cat((image, label), dim=1)
        
        latent = self.conv_layer(latent)
        latent = latent.view(-1, 512 * 4 * 4)
        output = self.fc_layer(latent)
        
        return output.squeeze()

if __name__ == "__main__":    
    model = DecompGen(5, 100, 10)
    noise = torch.randn(16, 100)
    label = torch.randint(0, 10, (16,))
    
    output = model(noise, label)
    
    print(output.shape)
    
    model = BaseGen(100, 10)
    
    output = model(noise, label)
    
    print(output.shape)
    
    model = Disc(10)
    image = output
    label = torch.randint(0, 10, (16,))
    
    output = model(image, label)
    
    print(output.shape)
