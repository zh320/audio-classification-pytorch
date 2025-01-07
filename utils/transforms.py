import torch
import numpy as np


def batch_mixup(x, y, alpha=1.0):
    ''' 
        paper: mixup: Beyond Empirical Risk Minimization
        link: https://arxiv.org/abs/1710.09412
    '''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
