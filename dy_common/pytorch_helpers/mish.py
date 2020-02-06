"""
mish (https://arxiv.org/abs/1908.08681) implementation

Code from https://github.com/digantamisra98/Mish
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    if torch.cuda.is_available():
        return cuda_mish(input)
    else:
        return cpu_mish(input)

@torch.jit.script
def cuda_mish(input):
    return input * torch.tanh(F.softplus(input))

@torch.jit.script
def cpu_mish(input):
    delta = torch.exp(-input)
    alpha = 1 + 2 * delta
    return input * alpha / (alpha + 2* delta * delta)

class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)