
import math
import os
from typing import List, Tuple, Any, Optional

import numpy as np
from torch.utils.data import DataLoader
import dataclasses as dtc
import torch.nn as nn
from torch import Tensor
import torch
import pytorch_lightning as pl
import torchvision as tv
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t
from torchvision.transforms.functional import crop
from torchvision.transforms import RandomHorizontalFlip, ColorJitter
from torchvision.transforms.functional import crop, resize
from matplotlib import image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset
from mimikit import FileWalker



def random_crop_resize(img, img_size):
    chans, img_w, img_h = img.shape
    max_shrink = 0.8
    pct_keep = np.random.rand() * (1 - max_shrink) + max_shrink
    out_h = int(img_h * pct_keep)
    out_w = int(img_w * pct_keep)
    # distort aspect
    out_h -= np.random.randint(16) if np.random.rand() < 0.5 else 0
    out_w -= np.random.randint(16) if np.random.rand() < 0.5 else 0
    free_h = img_h - out_h
    free_w = img_w - out_w
    crop_x = int((np.random.rand() + np.random.rand()) * free_w / 2)
    crop_y = int((np.random.rand() + np.random.rand()) * free_h / 2)
    out_img = crop(img, crop_x, crop_y, out_w, out_h)
    out_img = resize(out_img, img_size, antialias=True)
    w = out_img.size(1)
    h = out_img.size(2)
    if w > h:
        free = w - img_size
        crop_x = int((np.random.rand() + np.random.rand()) * free / 2)
        crop_y = 0
    else:
        free = h - img_size
        crop_y = int((np.random.rand() + np.random.rand()) * free / 2)
        crop_x = 0
    out_img = crop(out_img, crop_x, crop_y, img_size, img_size)
    return out_img


class AddNoise(object):
    def __init__(self, scale=0.1):
        self.scale = scale
        
    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * self.scale
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.scale)


class RandomCropResize(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, tensor):
        return random_crop_resize(tensor, self.size)
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)



# currently not used - just uses default weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# this version is unconditional - labels are not used in this one
class ImgsLabels(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 3
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 256x256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 128x128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf *2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, ign=None):
        return self.main(input).view(-1, 1)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        ngf = 64
        nz = latent_dim
        nc = 3
        # alternate between upsampled-conv and transposed-conv to try
        # to avoid checkerboard patterns.
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.1),
            # state size. 4 x 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.1),
            # state size. 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.1),
            # state size. 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.1),
            # state size. 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.1),
            # state size. 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.1),
            # state size. 128 x 128
            nn.ConvTranspose2d( ngf, ngf, 4, 2, 1, bias=False),
            nn.Conv2d(ngf, 3, (5, 5), padding=2),
            nn.Sigmoid()
            # state size. 256 x 256
        )

    def forward(self, input):
        return self.main(input)
    

img_size = 256
latent_dim = 64
device = 'cuda'
b1 =0.5
b2 = 0.999
lr = 0.0002

jitter_param = 0.1
lkrelu_negslop = 0.1
bn_eps = .1
bn_mom = .8
transforms = tv.transforms.Compose([
            RandomCropResize(img_size),
            ColorJitter(brightness=jitter_param,
                        contrast=jitter_param,
                        hue=jitter_param * 0.4,
                        saturation=jitter_param),
            AddNoise(0.006),
            RandomHorizontalFlip(0.5)
        ])


# load the images
paths = list(FileWalker('img', sources='./data/'))
dirs = [os.path.split(p)[0] for p in paths]
lbls = {lbl: i for i, lbl in enumerate(set(dirs))}
labels = torch.tensor([lbls[x] for x in dirs]).long()
images = [torch.as_tensor(image.imread(f)).transpose(0, 2).transpose(1, 2) / 256 for f in paths]
n_classes = len(list(lbls.keys()))


generator = Generator(latent_dim)
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

opt_g = torch.optim.Adam(generator.parameters(),
                         lr=lr, betas=(b1, b2))
opt_d = torch.optim.Adam(discriminator.parameters(),
                         lr=lr, betas=(b1, b2))

loss_fn = nn.BCELoss()



for epoch in range(4000):
    loss_d = 0.0
    loss_g = 0.0
    data = torch.stack([transforms(img) for img in images])
    dl = DataLoader(ImgsLabels(data.to(device), labels.to(device)), batch_size=24, shuffle=True, drop_last=True)
    for i, batch in enumerate(dl):
        #batch = next(iter(dl))
        real_imgs, labs = batch
        batch_size = real_imgs.shape[0]
        generator.zero_grad()
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, requires_grad=False, device=device)
        fake = torch.zeros(batch_size, 1, requires_grad=False, device=device)
        
        # Sample noise and labels as generator input
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        
        # Generate a batch of images
        gen_imgs = generator(z)
        
        discriminator.zero_grad()
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs)
        loss = loss_fn(validity, valid)    
        loss.backward()
        loss_g += loss
        opt_g.step()
        
        validity_real = discriminator(real_imgs)
        d_real_loss = loss_fn(validity_real, valid)
        validity_fake = discriminator(gen_imgs.detach())
        d_fake_loss = loss_fn(validity_fake, fake)
        
        # Total discriminator loss
        L = (d_real_loss + d_fake_loss) / 2
        loss_d += L
        opt_d.zero_grad()
        L.backward()
        opt_d.step()
    print(loss_g, loss_d)
    if epoch % 20 == 0:
        grid = tv.utils.make_grid(real_imgs[:16], 4)
        tv.utils.save_image(grid, f"images_output/train_{epoch}.jpeg")
        grid = tv.utils.make_grid(gen_imgs[:16], 4)
        tv.utils.save_image(grid, f"images_output/epcoch_{epoch}.jpeg")


#torch.save(generator.state_dict(), "cgan4-generator.states")
#torch.save(generator.state_dict(), "cgan4-discriminator.states")

