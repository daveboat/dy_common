"""
Some functions for visualzing image classifiers, using LIME
https://arxiv.org/abs/1602.04938
https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torch.nn.functional as F
from functools import partial

from lime import lime_image
from skimage.segmentation import mark_boundaries

from torchvision import transforms
from torchvision.models import resnet50

import json

def _get_input_transform():
    """
    A typical input transform for evaluation on models such as resnet and vgg, for testing purposes
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform

def _get_image(path):
    """
    Open an image with PIL and convert to RGB
    """
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def _batch_predict(images, model, device, input_transform):
    """
    Example of a batch predict function for feeding into lime's explainer. This function should take a batch of images
    as np arrays, feed them into the model, and return the batch of probabilities, and needs to be of the form

    batch_predict(images: iterable)

    Here we have a batch_predict with more inputs, so we can functools.partial it with the correct inputs

    This function can do minibatching, etc, if the number of input images is too large
    """
    batch = torch.stack(tuple(input_transform(i) for i in images), dim=0)

    batch = batch.to(device)

    with torch.no_grad():
        logits = model(batch)

    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def lime_visualize(imagepath, batch_predict, class_labels=None, top_labels=5, positive_only=False, num_features=5,
                   hide_rest=False, num_samples=1000):
    """
    imagepath: path to image
    batch_predict: A function which takes one argument, an iterable of images as numpy arrays, and returns a numpy
    array of batch probabilities. Basically batch_predict is what runs the model on the batch of input images, and is
    model-agnostic.
    class_names: A list of class labels, with the same order as the model's class indices.. If None, then only indices
    will be printed
    top_labels: Number of top labels to run
    positive_only: Only show positive contributions to label
    num_features: Number of image segments to show
    hide_rest: Whether or not to mask parts of image which are not positive or negative contributions
    num_samples: Number of random image patch samples of the image to generate. More samples means more accuracy, but
    more batches need to be put through batch_predict.
    """

    # load image from disk
    image = _get_image(imagepath)

    # initialize lime explainer
    explainer = lime_image.LimeImageExplainer()

    # run explaination
    explanation = explainer.explain_instance(np.array(image),
                                             batch_predict,  # classification function
                                             top_labels=top_labels,
                                             hide_color=0,
                                             num_samples=num_samples)  # number of images that will be sent to classification function

    for i in range(top_labels):
        label_index = explanation.top_labels[i]
        temp, mask = explanation.get_image_and_mask(label_index, positive_only=positive_only,
                                                    num_features=num_features, hide_rest=hide_rest)
        img_boundry1 = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry1)
        if class_labels is not None:
            plt.title('Classification contributions: Index {} ({})'.format(label_index, class_labels[label_index]))
        else:
            plt.title('Classification contributions: Index {}'.format(label_index))

        plt.show()

if __name__ == '__main__':
    # Typical usage for lime_visualize()

    imagepath = 'data/dog.jpeg'

    # get class labels
    with open('data/imagenet_class_index.json', 'r') as read_file:
        class_idx = json.load(read_file)
        class_labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

    # define model, transform, and device
    model = resnet50(pretrained=True)
    model.eval()
    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device('cpu')
    model.to(device)

    # define a partial function for batch_predict, so that we can pass it the model, device, and input transform
    batch_predict = partial(_batch_predict, model=model, device=device, input_transform=input_transform)

    lime_visualize(imagepath, batch_predict, class_labels=class_labels)