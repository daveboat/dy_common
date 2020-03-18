"""
Adaptive instance normalization, from https://github.com/edward-zhu/neural-transform

Lightly modified by me
"""

import torch
import torch.nn as nn


def get_mean_var(c):
    n_batch, n_ch, h, w = c.size()

    c_view = c.view(n_batch, n_ch, h * w)
    c_mean = c_view.mean(2)

    c_mean = c_mean.view(n_batch, n_ch, 1, 1).expand_as(c)
    c_var = c_view.var(2)
    c_var = c_var.view(n_batch, n_ch, 1, 1).expand_as(c)
    # c_var = c_var * (h * w - 1) / float(h * w)  # unbiased variance

    return c_mean, c_var


class AdaInstanceNormalization(nn.Module):
    '''
    Adaptive Instance Normalization Layer, from https://arxiv.org/pdf/1703.06868.pdf
    '''

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x_mean, x_var = get_mean_var(x)
        y_mean, y_var = get_mean_var(y)

        return y_var * (x - x_mean) / (x_var + self.eps) + y_mean


if __name__ == '__main__':
    AdaIn = AdaInstanceNormalization()

    x = torch.randn((2, 64, 256, 256))
    y = torch.randn((2, 64, 256, 256))

    out = AdaIn(x, y)

    print(out.size())
