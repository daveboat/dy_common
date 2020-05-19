"""
I3D model implemented in pytorch, from https://github.com/hassony2/kinetics_i3d_pytorch

I modified it to optionally use Mish activation instead of RELU, which I found works better when training from scratch.
When finetuning from the kinetics-400 pretrained weights, RELU should be used.

A note on the pretrained weights:
- They are pretrained on kinetics-400
- I've uploaded them to my Google drive, they can be downloaded here:
    rgb weights: https://drive.google.com/file/d/1jahRqrqVirpHyArAyYPkqID88r3LP9EI/view?usp=sharing
    flow weights: https://drive.google.com/file/d/1XvivdZpcWEmb6Jr2xzT_F4YKghw44VQx/view?usp=sharing
- Videos need to be preprocessed in a certain way for use with the pretrained weights (and I recommend you preprocess
in a similar way if you are training from scratch). The original paper did the following:
    1. Resize video frames so that the shorter side is 256 pixels
    2. For RGB frames, normalize from [0,255] to [-1,1]. For optical flow, use TVL1, and threshold values to [-20,20] by
    cutting off values larger than 20 and smaller than -20, then rescale to [-1,1]. For faster optical flow generation,
    I created a GPU-accelerated C++ OpenCV application for Linux which generates RGB and optical flow frames from a
    video. It can be found on my github here: https://github.com/daveboat/denseFlow_GPU. Since I was using this with
    I3D, the default options already do the 256-pixel resize and optical flow thresholding.
    3. A constant number of frames (N) needs to be chosen for training. The original paper chose 64 rgb frames and 64
    flow frames in the paper, and in their repo, use a 79-frame example. For training, they take the first N frames
    if the video is longer than N, or loop the video until N is reached if the video is shorter than N. (This is in
    contrast with more recent work, where generally the number of training frames is much smaller, and sampled randomly
    from the video during training. Using this method, usually a number, like 10, uniform samples are run through the
    model and have their results averaged at test-time.)
- At test-time, any number of frames greater than 8 can be used (though using a value close to the training value
works better). Note that, for both RGB and Flow, the feature vector (the output of the conv3d_0c_1x1 layer) is a 1x512
tensor for each 8 frames, so in general, you will get an (ceil(N/8))x512 size feature vector for an N-length input
sequence.
"""

import torch
from dy_common.pytorch_helpers.mish import mish


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), activation='relu',
                 padding='SAME', use_bias=False, use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError('padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d( in_channels, out_channels, kernel_size, stride=stride, padding=pad_size,
                                               bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding_shape, stride=stride,
                                          bias=use_bias)
        else:
            raise ValueError('padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'mish':
            self.activation = mish

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        # if self.activation is not None:
        #     out = torch.nn.functional.relu(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1, 1, 1), activation=activation)

        # Branch 1
        branch_1_conv1 = Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1), activation=activation)
        branch_1_conv2 = Unit3Dpy(out_channels[1], out_channels[2], kernel_size=(3, 3, 3), activation=activation)
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1), activation=activation)
        branch_2_conv2 = Unit3Dpy(out_channels[3], out_channels[4], kernel_size=(3, 3, 3), activation=activation)
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1), activation=activation)
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    """
    activation: activation used for all layers except final layer before softmax. Default='relu'
    final_activation: final layer activation, before softmax. Should leave as None
    """
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 activation='relu',
                 final_activation=None,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError('{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(out_channels=64,in_channels=in_channels,kernel_size=(7, 7, 7),stride=(2, 2, 2),
                                 padding='SAME', activation=activation)
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1), padding='SAME',
                                 activation=activation)
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(out_channels=192,in_channels=64,kernel_size=(3, 3, 3),padding='SAME',
                                 activation=activation)
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32], activation=activation)
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64], activation=activation)

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64], activation=activation)
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64], activation=activation)
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64], activation=activation)
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64], activation=activation)
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128], activation=activation)

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128], activation=activation)
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128], activation=activation)

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                      activation=final_activation, use_bias=True, use_bn=False)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inp):
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits