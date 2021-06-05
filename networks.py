import dataclasses as dtc
import torch.nn as nn
import torch
import numpy as np
from inspect import signature
import pytorch_lightning as pl


###################
# Helper
###################

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


def filter_cls_kwargs(cls, kwargs):
    valids = signature(cls.__init__).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valids}

###################
# Inputs Generators
###################


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class NoiseAndEmbeddings(nn.Module):
    """
    concatenate noise and embeddings
    """
    noise_dim: int = 32
    emb_dim: int = 32
    normalize: bool = True
    n_classes: int = 10

    def __post_init__(self):
        nn.Module.__init__(self)
        self.labels_embs = nn.Embedding(self.n_classes, self.emb_dim)

    def forward(self, labels):
        embs = self.labels_embs(labels)
        z = torch.randn(embs.size(0), self.noise_dim).to(embs)
        if self.normalize:
            # normalize z to embs' distribution...
            z = embs.std(dim=-1, keepdim=True) * z + embs.mean(dim=-1, keepdim=True)
        return torch.cat((z, embs), dim=-1)


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class ParametrizegGaussian(nn.Module):
    """
    parametrize gaussian noise with embeddings
    """
    latent_dim: int = 64
    n_classes: int = 10

    def __post_init__(self):
        nn.Module.__init__(self)
        self.labels_mu = nn.Embedding(self.n_classes, self.latent_dim)
        self.labels_sigma = nn.Embedding(self.n_classes, self.latent_dim)

    def forward(self, labels):
        batch_size, device = labels.size(0), labels.device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        mu, sigma = self.labels_mu(labels), self.labels_sigma(labels)
        return z * sigma.mul_(.5).exp_() + mu

    def entropy(self):
        pi = torch.acos(torch.zeros(1)).item()
        return (.5 + .5 * torch.log(2 * pi * self.labels_sigma.weight.exp())).sum()


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class MultinomialGenerator(nn.Module):
    """
    parametrize a multinomial distribution with embeddings
    """
    latent_dim: int = 64
    n_classes: int = 10
    count: int = 100

    def __post_init__(self):
        nn.Module.__init__(self)
        self.labels_probs = nn.Embedding(self.n_classes, self.latent_dim)

    def forward(self, labels):
        embs = self.labels_probs(labels)
        probs = nn.Softmax(dim=-1)(embs)
        z = torch.distributions.Multinomial(self.count, probs).sample()
        return z * probs / self.count

    def entropy(self):
        pr = nn.Softmax(dim=-1)(self.labels_probs.weight)
        return - (torch.log(pr) * pr).sum()


###############
# Layer Helpers
###############


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class LinearLayer(nn.Module):
    in_feat: int = 64
    out_feat: int = 64
    lkrelu_negslop: float = .02
    normalize: bool = True
    bn_eps: float = .1
    bn_mom: float = .8
    dropout: float = 0.

    def __post_init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(self.in_feat, self.out_feat)
        if self.normalize:
            self.bn = nn.BatchNorm1d(self.out_feat, self.bn_eps, self.bn_mom)
        else:
            self.bn = lambda x: x
        self.relu = nn.LeakyReLU(self.lkrelu_negslop, inplace=True)
        self.dp = nn.Dropout(self.dropout) if self.dropout > 0. else lambda x: x

    def forward(self, x):
        return self.dp(self.relu(self.bn(self.fc(x))))


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class Conv2dLayer(nn.Module):
    in_feat: int = 64
    out_feat: int = 64
    transpose: bool = False
    kernel: [tuple, int] = (3,)
    padding: [tuple, int] = (1,)
    stride: [tuple, int] = (1,)
    dilation: [tuple, int] = (1,)
    bias: bool = True
    lkrelu_negslop: float = .2
    normalize: bool = True
    bn_eps: float = .0001
    bn_mom: float = .1
    dropout: float = 0.
    upsample: bool = False

    def __post_init__(self):
        nn.Module.__init__(self)
        mod = nn.ConvTranspose2d if self.transpose else nn.Conv2d
        self.cv = mod(self.in_feat, self.out_feat,
                      self.kernel, self.stride, self.padding, self.dilation, bias=self.bias)
        self.relu = nn.LeakyReLU(self.lkrelu_negslop, inplace=True)
        if self.normalize:
            self.bn = nn.BatchNorm2d(self.out_feat, self.bn_eps, self.bn_mom)
        else:
            self.bn = lambda x: x
        self.dp = nn.Dropout2d(self.dropout)
        self.up = nn.Upsample(2) if self.upsample else lambda x: x

    def forward(self, x):
        return self.up(self.dp(self.relu(self.bn(self.cv(x)))))


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class ImgNet(pl.LightningModule):
    img_channels: int = 3
    img_size: int = 32
    n_classes: int = 10

    img_shape = property(lambda self: (self.img_channels, self.img_size, self.img_size))
    flat_img_dim = property(lambda self: int(np.prod(self.img_shape)))

    def inpt_(self):
        return nn.Module()

    def layers_(self):
        return nn.Module()

    def outpt_(self):
        return nn.Module()

    def __post_init__(self):
        pl.LightningModule.__init__(self)
        self.inpt = self.inpt_()
        self.layers = self.layers_()
        self.outpt = self.outpt_()
        self.model = nn.Sequential(
            self.inpt,
            self.layers,
            self.outpt
        )

    def forward(self, x):
        return self.model(x)

