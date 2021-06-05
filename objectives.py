import torch
import torch.nn as nn

"""
- Always set net.automatic_optimization to False

- G must have a method gen_input(self, labels) -> z
- G.forward(z, gen_labels) -> image
- D.forward(image) -> predictions

"""


def real_vs_fake(net, batch, batch_idx, optimizer_idx):
    """
    have D output 1 scalar 0 <= x <= 1
    """
    real_imgs, labels = batch
    batch_size = real_imgs.shape[0]
    g_opt, d_opt = net.optimizers()
    d_opt.zero_grad()
    g_opt.zero_grad()
    criterion = nn.BCELoss()
    n_classes, device = net.n_classes, net.device

    # Sample noise and labels as generator input
    gen_labels = torch.randint(n_classes, (batch_size,), device=device)
    reals = torch.ones(batch_size, device=device)
    fakes = torch.zeros(batch_size, device=device)

    # Generate a batch of images
    gen_imgs = net.generator(gen_labels)

    rv = dict()

    # TRAIN D
    validity_real = net.discriminator(real_imgs)
    d_real_loss = criterion(validity_real, reals)

    validity_fake = net.discriminator(gen_imgs.detach())
    # fake classes are offset, hence gen_labels + n_classes
    d_fake_loss = criterion(validity_fake, fakes)

    L = (.5 * d_fake_loss + .5 * d_real_loss)
    rv['D_loss'] = L
    rv['Dr_loss'] = d_real_loss
    rv['Df_loss'] = d_fake_loss

    # step
    L.backward()
    d_opt.step()

    # train generator
    validity = net.discriminator(gen_imgs)
    L = criterion(validity, reals)
    rv["Gr_loss"] = L
    L.backward()
    g_opt.step()
    net.log_dict(rv, prog_bar=True, logger=True, on_step=True)
    return rv


def real_vs_fake_stats(net, batch, batch_idx, optimizer_idx):
    """
    have D output N scalar 0 <= x <= 1
    """
    real_imgs, labels = batch
    batch_size = real_imgs.shape[0]
    g_opt, d_opt = net.optimizers()
    d_opt.zero_grad()
    g_opt.zero_grad()
    criterion = nn.BCELoss()
    n_classes, device = net.n_classes, net.device

    # Sample noise and labels as generator input
    gen_labels = torch.randint(n_classes, (batch_size,), device=device)
    reals = torch.ones(batch_size, device=device)
    fakes = torch.zeros(batch_size, device=device)

    # Generate a batch of images
    gen_imgs = net.generator(gen_labels)

    rv = dict()

    # TRAIN D

    validity_real = net.discriminator(real_imgs)
    d_real_loss = criterion(validity_real.mean(dim=-1), reals)
    d_real_std = validity_real.std(dim=-1).mean()

    validity_fake = net.discriminator(gen_imgs.detach())
    d_fake_loss = criterion(validity_fake.mean(dim=-1), fakes)
    d_fake_std = validity_fake.std(dim=-1).mean()

    L = (.5 * d_fake_loss + .5 * d_real_loss) + (.5 * d_real_std + .5 * d_fake_std)
    rv['D_loss'] = L
    rv['Dr_loss'] = d_real_loss
    rv['Df_loss'] = d_fake_loss
    rv['Dr_std'] = d_real_std
    rv['Df_std'] = d_fake_std

    # step
    L.backward()
    d_opt.step()

    # TRAIN G

    validity = net.discriminator(gen_imgs)
    L = criterion(validity.mean(dim=-1), reals)
    g_real_std = validity.std(dim=-1).mean()
    rv["Gr_loss"] = L
    rv['Gr_std'] = g_real_std
    L.backward()
    g_opt.step()

    net.log_dict(rv, prog_bar=True, logger=True, on_step=True)
    return rv


def wasserstein_loss(net, batch, batch_idx, optimizer_idx):
    """
    Have D output 1 (unconstrained) scalar
    """
    real_imgs, labels = batch
    batch_size = real_imgs.shape[0]
    g_opt, d_opt = net.optimizers()
    d_opt.zero_grad()
    g_opt.zero_grad()
    n_classes, device = net.n_classes, net.device

    # Sample noise and labels as generator input
    gen_labels = torch.randint(n_classes, (batch_size,), device=device)
    gen_imgs = net.generator(gen_labels)

    rv = dict()

    # TRAIN D

    validity_real = net.discriminator(real_imgs)
    d_real_loss = - validity_real.mean()

    validity_fake = net.discriminator(gen_imgs.detach())
    d_fake_loss = validity_fake.mean()

    L = d_fake_loss + d_real_loss
    rv['D_loss'] = L
    rv['Dr_loss'] = d_real_loss
    rv['Df_loss'] = d_fake_loss

    # step
    L.backward()
    d_opt.step()

    # TRAIN G

    validity = net.discriminator(gen_imgs)
    L = - validity.mean()
    rv['Gr_loss'] = L
    L.backward()
    g_opt.step()

    net.log_dict(rv, prog_bar=True, logger=True, on_step=True)
    return rv


def real_cls_fake_cls(net, batch, batch_idx, optimizer_idx):
    """
    have D output logits for Pr ( n_classes * 2 )
    """
    real_imgs, labels = batch
    batch_size = real_imgs.shape[0]
    g_opt, d_opt = net.optimizers()
    d_opt.zero_grad()
    g_opt.zero_grad()
    criterion = nn.CrossEntropyLoss()
    n_classes, device = net.n_classes, net.device

    # Sample noise and labels as generator input
    gen_labels = torch.randint(n_classes, (batch_size,), device=device)

    # Generate a batch of images
    gen_imgs = net.generator(gen_labels)

    rv = dict()

    # TRAIN D

    validity_real = net.discriminator(real_imgs)
    d_real_loss = criterion(validity_real, labels)

    validity_fake = net.discriminator(gen_imgs.detach())
    # fake classes are offset, hence gen_labels + n_classes
    d_fake_loss = criterion(validity_fake, gen_labels + n_classes)

    L = (.5 * d_fake_loss + .5 * d_real_loss)
    rv['loss'] = L
    rv['D_loss'] = L
    rv['Dr_loss'] = d_real_loss
    rv['Df_loss'] = d_fake_loss

    # step
    L.backward()
    d_opt.step()

    # TRAIN G

    validity = net.discriminator(gen_imgs)
    L = criterion(validity, gen_labels)
    rv['loss'] = L
    rv["Gr_loss"] = L
    L.backward()
    g_opt.step()

    net.log_dict(rv, prog_bar=True, logger=True, on_step=True)
    return rv


def fast_gradient_sign_step(gen_imgs, d_net, d_opt, criterion, reals, fakes):
    d_opt.zero_grad()
    adv_inpt = torch.tensor(gen_imgs.detach(), requires_grad=True)
    val = d_net(adv_inpt)
    adv_L = criterion(val, reals)
    adv_L.backward()
    adv_inpt = adv_inpt - torch.sign(adv_inpt.grad) * 1e-1
    d_opt.zero_grad()

    val2 = d_net(adv_inpt)
    true_L = criterion(val2, fakes)
    return true_L
