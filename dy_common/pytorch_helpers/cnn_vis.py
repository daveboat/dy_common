import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import alexnet
from .gaussiansmoothing import GaussianSmoothing


def cnn_visualize(model, device, class_index, input_height=224, input_width=224, channels=3, iterations=5000, lr=0.1,
                  regularization=0.1, smoothing_interval=250, smoothing_kernel_size=5, smoothing_std=1,
                  value_threshold_interval=100, value_threshold=0.01):
    """
    A function which uses gradient ascent to find the input image which gives the highest probability of the output
    being class_index

    Periodically sets gradients and values of the image to zero which are below a threshold, also periodically
    Gaussian smoothes the image.
    """

    # What we want to do is the following:
    #   - For a given output logit, calculate the input tensor which makes that logit the largest.
    #   - To do this, we use gradient ascent (i.e. gradient descent with a negative objective function) on the output
    #     logit with back-propagated gradients of the input tensor
    #   - We do a forward pass to calculate the output logit, calculate the loss = - logit + L2(input image), then we
    #     backprop to get the gradients at the input, and either manually descend or use an optimizer
    #   - Periodically, we should clip values, gradients, and smooth the image?

    # some assertion checks on the input parameters
    assert smoothing_kernel_size % 2 != 0, 'Smoothing kernel size must be odd. Got {}'.format(smoothing_kernel_size)

    # make smoother
    smoother = GaussianSmoothing(channels, smoothing_kernel_size, smoothing_std)
    smoother = smoother.to(device)
    smoother_pad = smoothing_kernel_size // 2

    # send model to device
    model = model.to(device)

    # We want to use gradient ascent to get the input image which maximizes a particular output logit.
    # Start by turning the model itself into eval mode, since we don't care about its gradients, only the gradients of
    # the input
    model.eval()

    # Then, we create a random or zeroed input image, of size (1, C, H, W), with requires_grad turned on
    image = torch.randn((1, channels, input_height, input_width), requires_grad=True, device=device)

    # Now, we define an optimizer which will handle gradient descent on the input image's gradients
    optimizer = torch.optim.SGD([image], lr=lr)

    for i in range(iterations):
        # zero gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(image)

        # make sure

        # compute loss. We assume the loss will have shape [1, num_classes]. Also add L2 regularization.
        loss = - logits[0][class_index] + regularization * torch.norm(image.view((-1,)), p=2)

        # backward pass
        loss.backward()

        if i % smoothing_interval == 0 and i > 0:
            print('smoothing...')
            image_pad = F.pad(image.clone(), (smoother_pad, smoother_pad, smoother_pad, smoother_pad), mode='reflect')
            image.data = smoother(image_pad)
        if i % value_threshold_interval == 0 and i > 0:
            print('clipping values...')
            image_min = torch.min(image)
            image.data[image - image_min < value_threshold] = image_min

        # update image
        optimizer.step()

        print('Iteration {}, loss: {}'.format(i, loss))

    # return the image as a numpy array, in numpy's expected [H, W, C] format
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, (1, 2, 0))
    imgmin, imgmax = np.min(image), np.max(image)
    image = (image - imgmin)/(imgmax - imgmin)
    return image

if __name__ == '__main__':
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    # Choose a pretrained model
    model = alexnet(pretrained=True)  # alexnet input size is 256x256

    image = cnn_visualize(model, device, class_index=100, input_height=256, input_width=256, channels=3)

    plt.imshow(image)
    plt.show()
