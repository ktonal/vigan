"""

WORKS on the mimikit branch experiment/cgan

"""

import dataclasses as dtc
import torch.nn as nn
import torch

import pytorch_lightning as pl

import torchvision as tv
import os

from mimikit.data import Feature
import sys

# this allows import from vigan/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from vigan.sampling import *
from vigan.objectives import *
from vigan.networks import *


# Sketch for an Image Loader
@dtc.dataclass(unsafe_hash=True)
class Image(Feature):
    __ext__ = 'img'

    img_size: int = 128

    @property
    def encoders(self):
        transforms = tv.transforms.Compose([
            tv.transforms.Resize(self.img_size),
            tv.transforms.CenterCrop(self.img_size),
        ])
        return {torch.Tensor: transforms}

    def load(self, path):
        img = tv.io.read_image(path).to(torch.float32)
        # make the images look like a sigmoid (a fake!) by dividing by 255
        img = self.encode(img).unsqueeze(0).cpu().numpy() / 255
        return img


@dtc.dataclass(unsafe_hash=True)
class DirectoryLabels(Feature):
    __ext__ = 'img'

    def load(self, path):
        # one image = one placeholder (paths are automatically saved in db.files)
        return np.array([1])

    def post_create(self, db, schema_key):
        # once we know all the paths -> converts them to set indices
        feat = getattr(db, schema_key)
        paths = [os.path.split(p)[0] for p in feat.files.name.values]
        lbls = {lbl: i for i, lbl in enumerate(set(paths))}
        return np.array([lbls[x] for x in paths])


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False)
class CGAN(pl.LightningModule):

    # DB
    @classmethod
    def schema(cls, img_size=64):
        return {
            'img': Image(img_size=img_size),
            'labels': DirectoryLabels()
        }

    # Dataset
    def batch_signature(self, stage='fit'):
        from mimikit.data import Input
        return Input('img', transform=tv.transforms.RandomCrop(self.img_size, padding=self.rcrop,
                                                               padding_mode='reflect')), Input('labels')

    # Dataloader
    def loader_kwargs(self, stage, datamodule):
        return dict(
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

    # __init__
    img_channels: int = 3
    img_size: int = 128
    rcrop: int = 0
    n_classes: int = 10
    latent_dim: int = 64
    n_layers: int = 4
    lkrelu_negslop: float = .2
    bn_eps: float = .1
    bn_mom: float = .8

    lr: float = 2e-4
    b1: float = .5
    b2: float = .999

    batch_size: int = 8

    sample_every: int = 100

    def generator_(self):
        gan = self

        class G(ImgNet):

            def inpt_(self):
                return MultinomialGenerator(
                    latent_dim=gan.latent_dim,
                    n_classes=gan.n_classes
                )

            def layers_(self):
                nz = gan.latent_dim
                dims = [nz * 2 ** i for i in range(gan.n_layers)]
                return nn.Sequential(
                    *[LinearLayer(
                        dims[i], dims[i + 1],
                        normalize=True,
                        # pass matching kwargs stored in gan
                        **filter_cls_kwargs(LinearLayer, dtc.asdict(gan))
                    )
                        for i in range(gan.n_layers - 1)],
                    nn.Linear(dims[-1], self.flat_img_dim)
                )

            def outpt_(self):
                return nn.Sequential(
                    nn.Hardsigmoid(),
                    Reshape(self.img_shape)
                )

        return G(self.img_channels, self.img_size, self.n_classes)

    def discriminator_(self):
        gan = self

        class D(ImgNet):

            def inpt_(self):
                return nn.Flatten()

            def layers_(self):
                dims = [gan.latent_dim * 2 ** i for i in range(gan.n_layers - 1, -1, -1)]
                self.dims = dims
                return nn.Sequential(
                    nn.Linear(self.flat_img_dim, dims[0]),
                    *[LinearLayer(
                        dims[i], dims[i + 1],
                        normalize=False,
                        # pass matching kwargs stored in gan
                        **filter_cls_kwargs(LinearLayer, dtc.asdict(gan))
                    )
                        for i in range(gan.n_layers - 2)],
                )

            def outpt_(self):
                return nn.Sequential(
                    nn.Linear(self.dims[-1] * 2, 1),
                    # nn.Sigmoid()
                )

        return D(self.img_channels, self.img_size, self.n_classes)

    def __post_init__(self):
        pl.LightningModule.__init__(self)
        self.generator = self.generator_()
        self.discriminator = self.discriminator_()
        self.automatic_optimization = False
        self._set_hparams(dtc.asdict(self))

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))

        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        return wasserstein_loss(self, batch, batch_idx, optimizer_idx)

    def sample_image(self, n_cols):
        """sample a grid of generated digits ranging from 0 to n_classes"""
        self.generator.eval()
        # Get labels ranging from 0 to n_classes for n rows
        labels = torch.tensor([num for _ in range(n_cols) for num in range(self.n_classes)],
                              device=self.device, dtype=torch.long)
        imgs = self.generator.forward(labels)
        self.generator.train()
        return imgs

    def on_train_batch_end(self, *args, **kwargs):
        """
        Save a grid of outputs
        """

        if self.global_step == 1:
            # reset the output folder...
            shutil.rmtree("images_output/")
            os.makedirs("images_output/")
        if self.global_step % self.sample_every == 0:
            if not os.path.isdir('images_output'):
                os.makedirs('images_output')
            imgs = self.sample_image(self.n_classes)
            grid = tv.utils.make_grid(imgs, self.n_classes)
            tv.utils.save_image(grid, f"images_output/step_{self.global_step}.jpeg")


if __name__ == '__main__':
    # for a notebook or as cli
    import mimikit as mmk
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import shutil

    img_size = 128

    db = mmk.Database.create('img_test.h5', ['./data'],
                             CGAN.schema(img_size=img_size))

    n_classes = len(set(db.labels[:]))

    net = CGAN(
        sample_every=200,
        img_size=img_size,
        n_classes=n_classes,
        latent_dim=128,
        n_layers=4,
        rcrop=0,
        lkrelu_negslop=.2,
        bn_eps=0.0001,
        bn_mom=0.05,
        batch_size=n_classes * 2,
        lr=3e-4,
        b1=.5, b2=.995,

    )

    print(net.hparams)

    print(net)

    # this class knows how to instantiate db, Dataset and DataLoader from the model
    dm = mmk.DataModule(net, db)

    trainer = mmk.get_trainer(max_epochs=30000,
                              enable_pl_optimizer=False,
                              callbacks=[],
                              checkpoint_callback=False)

    if os.path.exists("logs/"):
        shutil.rmtree("logs/")

    trainer.fit(net, datamodule=dm)

    # plot the losses
    losses = pd.read_csv('./logs/metrics.csv')
    losses.drop(labels=['epoch', 'created_at'], axis=1).plot()
    plt.show()
