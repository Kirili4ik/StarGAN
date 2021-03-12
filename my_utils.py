import random
import numpy as np
import torch
from torch import nn

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), requires_grad=False, device=device), # gradients w.t. output. 1 is default
        create_graph=True,
        retain_graph=True, # keep all gradients for further optimization steps
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def generate_target_label(batch_size, n_domains, device):
    target_label = torch.randint(high=2, size=(batch_size, n_domains-3), device=device).type(torch.float)
    target_label = torch.cat((torch.zeros(size=(batch_size, 3), device=device),
                              target_label), dim=1)

    hairs = torch.ones(size=(batch_size, 1), device=device).squeeze(1).type(torch.float)
    insert_rows = hairs == 1

    colour_nums = torch.randint(high=3, size=(insert_rows.sum(), 1), device=device).squeeze(1)
    
    target_label[insert_rows, colour_nums] = 1
    return target_label