import torch
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas

def get_alpha_cumprod(betas):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod

def noise_boxes(gt_boxes, t, alphas_cumprod):
    """
    Add noise to boxes at timestep t.
    """
    sqrt_alpha_cumprod_t = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_cumprod_t = (1 - alphas_cumprod[t]) ** 0.5
    noise = torch.randn_like(gt_boxes)
    noisy_boxes = sqrt_alpha_cumprod_t * gt_boxes + sqrt_one_minus_alpha_cumprod_t * noise
    return noisy_boxes
