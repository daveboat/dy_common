### Some notes on the I3D model

This folder has a pytorch implementation of I3D (`i3d.py`), copied shamelessly from [https://github.com/hassony2/kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch). 
The original repo and paper are [here](https://github.com/deepmind/kinetics-i3d).

I modified it to optionally use Mish activation instead of RELU, which I found works better when training from scratch.
When finetuning from the kinetics-400 pretrained weights, RELU should be used.

A note on the pretrained weights and the train/test pipeline:
- The pretrained weights are pretrained on kinetics-400
- I've uploaded the weights to my Google drive, they can be downloaded here:
    rgb weights: [https://drive.google.com/file/d/1jahRqrqVirpHyArAyYPkqID88r3LP9EI/view?usp=sharing](https://drive.google.com/file/d/1jahRqrqVirpHyArAyYPkqID88r3LP9EI/view?usp=sharing)
    flow weights: [https://drive.google.com/file/d/1XvivdZpcWEmb6Jr2xzT_F4YKghw44VQx/view?usp=sharing](https://drive.google.com/file/d/1XvivdZpcWEmb6Jr2xzT_F4YKghw44VQx/view?usp=sharing)
- Videos need to be preprocessed in a certain way for use with the pretrained weights (and I recommend you preprocess
in a similar way if you are training from scratch). The original paper did the following:
    1. Resize video frames so that the shorter side is 256 pixels. At train-time, a random 224x224 crop is taken, and at
    test time, a center 224x224 crop is taken.
    2. For RGB frames, normalize from [0,255] to [-1,1]. For optical flow, use TVL1, and threshold values to [-20,20] by
    cutting off values larger than 20 and smaller than -20, then rescale to [-1,1]. For faster optical flow generation,
    I created a GPU-accelerated C++ OpenCV application for Linux which generates RGB and optical flow frames from a
    video. It can be found on my github [here](https://github.com/daveboat/denseFlow_GPU). Since I was using this with
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

Finally, I built `videodataset.py` for the more recent style of video train/test pipelines, where small numbers of
frames are sampled from videos. You'll need to craft your own dataloader for I3D. If you're using my dense optical flow
application, this means loading and stacking images into RGB and Flow tensors.