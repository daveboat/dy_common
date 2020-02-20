"""
Implementations of Center Loss and Contrastive Center Loss, from https://ydwen.github.io/papers/WenECCV16.pdf and
https://arxiv.org/pdf/1707.07391.pdf

Most code from https://github.com/lyakaap/image-feature-learning-pytorch, updated here for pytorch >= 1.4

The way center loss works is, we include a loss of the form

L_c = sum_i||x_i - c_i||

where x_i is the feature vector for the i-th class, and c_i is the center of feature vectors for the i-th class. x_i
refers to feature vectors, which, in practice, are the neural network's outputs before the final classification FC
layer.

Theoretically, we could go through the training set and compute the centers c_i by averaging each class. However, in
practice, this usually isn't feasible, so we make the centers a learnable parameter, and adjust it per batch. This means
that, during training, there needs to be a separate SGD optimizer, with learning rate alpha ~= 0.5, which handles only
the centerloss.centers parameter. During evaluation, eval() will freeze the centers so that the loss can be computed
without propagating gradients. This also means that checkpoints need to include the state_dict of the center loss
object.

Center loss is usually added to a classification loss such as cross-entropy or focal loss, like

L = L_CE + lambda * L_c

and promotes class compactness in feature space, since it punishes class feature vectors which are too far from the
center of that class's features.

Contrastive center loss is a variation of center loss which, in addition to promoting class compactness (punishing L2
distance away from class centers), also promotes inter-class separation, by considering distance between x_i and all
other class centers.
"""

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0):
        super(CenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))

    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.index_select(dim=0, index=y)
        intra_distances = hidden.dist(expanded_centers)  # torch tensor.dist() is (by default) l2 distance
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss

class ContrastiveCenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))

    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, y.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)  # this sums the inter-class distances, and then subtracts the intra-class distance, as in the denominator of https://arxiv.org/pdf/1707.07391.pdf equation (3)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / (inter_distances + epsilon) / 0.1
        return loss

def test():
    ct = CenterLoss(2, 2)
    y = torch.LongTensor([0, 0, 0, 1])
    feat = torch.randn((4, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()

def test_contrastive():
    ct = ContrastiveCenterLoss(2, 3)
    y = torch.LongTensor([0, 0, 0, 1, 1, 2])
    feat = torch.randn((6, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()

if __name__ == '__main__':
    test()
    test_contrastive()