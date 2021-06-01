import torch


def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=-1, keepdim=True)
    high_norm = high/torch.norm(high, dim=-1, keepdim=True)
    eps = torch.tensor(1e-8).to(high)
    omega = torch.acos(torch.clamp((low_norm*high_norm).sum(dim=-1), min=-1+eps, max=1-eps))
    so = torch.sin(omega)
    out = (torch.sin((1.0-val)*omega)/so).unsqueeze(-1)*low + (torch.sin(val*omega)/so).unsqueeze(-1) * high
    return out


def slerp_space(low, high, n_steps):
    return slerp(torch.linspace(0., 1., n_steps), low, high)

