"""
A video dataset for pytorch, using opencv to load and rescale the videos
"""

import math
import os
import torch
import random
import cv2
import numpy as np


def uniform_sample(video_tensor, sample_len, num_samples):
    """
    Given a 5D video tensor (NCDHW), take num_samples samples of sample_len length, distributed uniformly from beginning
    to end, keeping all other dimensions the same

    Returns a list of sampled clips
    """
    # get the frame length of the tensor, which is dimension 2
    video_len = video_tensor.size(2)

    # this simple algorithm runs into trouble if I have num_samples or sample_len >= length of the video tensor, so
    # we should check for this
    assert video_len > sample_len and video_len > num_samples, \
        'uniform_sample(): Video length ({}) should be greater than both sample length ({}) and number of samples ({})'.format(video_len, sample_len, num_samples)

    start = max(video_len // (num_samples + 1) - math.ceil(sample_len / 2), 0)
    stride = round((video_len - start) / (num_samples + 1))
    overshoot = start + (num_samples - 1) * stride + sample_len - video_len
    if overshoot > 0:
        stride -= math.ceil(overshoot/(num_samples - 1))

    outlist = []
    for i in range(num_samples):
        clip = video_tensor[:, :, start + i * stride:start + i * stride + sample_len, :, :]

        # this should always pass as long as video_len > sample_len and video_len > num_samples, but check anyways
        if clip.size(2) == sample_len:
            outlist.append(clip)

    return outlist


def uniform_crop(video_tensor, crop_size, num_crops):
    """
    Given a 5D video tensor (NCDHW), take num_crops crops of square size crop_size*crop_size, uniformly along the longer
    spatial dimension. If num_crops == 1, a center crop should be taken
    """
    # get the height and width of the video
    video_height = video_tensor.size(3)
    video_width = video_tensor.size(4)

    # make sure crop size is not equal or larger than the smaller of the two video dimensions
    assert crop_size <= min(video_height, video_width), \
        'uniform_crop(): crop size ({}) must be smaller than or equal to both video width ({}) and video height ({})'.format(crop_size, video_width, video_height)

    # make crops for when h > w and for when w > h
    outlist = []
    if video_height > video_width:  # in this situation, we sample along height and center crop along width
        start = max(video_height // (num_crops + 1) - math.ceil(crop_size / 2), 0)
        stride = round((video_height - start) / (num_crops + 1))
        overshoot = start + (num_crops - 1) * stride + crop_size - video_height
        if overshoot > 0:
            stride -= math.ceil(overshoot/(num_crops - 1))

        for i in range(num_crops):
            clip = video_tensor[:, :, :, start + i * stride:start + i * stride + crop_size, int((video_width - crop_size)/2):int((video_width - crop_size)/2) + crop_size]
            outlist.append(clip)

    else:  # in this situation, we sample along width and center crop along height. If height and width are equal, default to sampling along width since more interesting details tend to be along the landscape direction
        start = max(video_width // (num_crops + 1) - math.ceil(crop_size / 2), 0)
        stride = round((video_width - start) / (num_crops + 1))
        overshoot = start + (num_crops - 1) * stride + crop_size - video_width
        if overshoot > 0:
            stride -= math.ceil(overshoot/(num_crops - 1))

        for i in range(num_crops):
            clip = video_tensor[:, :, :, int((video_height - crop_size) / 2):int((video_height - crop_size) / 2) + crop_size, start + i * stride:start + i * stride + crop_size]
            outlist.append(clip)

    return outlist


def random_sample(frames, sample_length):
    """
    Samples a random sequence of sample_length frames from frames. Frames are assumed to be in DHWC order
    """

    frame_length = frames.shape[0]

    # if, for some reason, we request more samples than there are frames, just return the entire video, with a
    # warning
    if sample_length >= frame_length:
        print('Warning - VideoDataset._random_sample(): sample_length ({}) >= frame_length ({})'.format(sample_length,
                                                                                                        frame_length))
        return frames
    else:
        start_frame = random.randint(0, frame_length - sample_length)

        return frames[start_frame:start_frame + sample_length, :, :, :]


def random_crop(frames, crop_size):
    """
    Perform a random crop of the video of crop_size x crop_size. Frames are assumed to be in DHWC order
    """

    # get frame height and width. The frame is in order [1, frames, height, width, channels]
    frame_height = frames.shape[1]
    frame_width = frames.shape[2]

    # randomly crop by choosing a number from 0 to frame_height/width - 224
    height_start = random.randint(0, frame_height - crop_size)
    width_start = random.randint(0, frame_width - crop_size)

    return frames[:, height_start:height_start + crop_size, width_start:width_start + crop_size, :]


def center_crop(frames, crop_size):
    """
    Perform a center crop of the video of crop_size x crop_size. Frames are assumed to be in DHWC order
    """
    # get frame height and width. The frame is in order [frames, height, width, channels]
    frame_height = frames.shape[1]
    frame_width = frames.shape[2]

    # return a center crop
    return frames[:, :, int((frame_height - crop_size)/2):int((frame_height - crop_size)/2) + crop_size, int((frame_width - crop_size)/2):int((frame_width - crop_size)/2) + crop_size, :]


def horizontal_flip(frames):
    """
    Perform a horizontal flip of the video. Frames are assumed to be an np.array in DHWC order
    """

    # a horizontal flip corresponds to reversing indices on the width index
    # need to do the extra operation because torch.from_numpy doesn't support negative strides (i.e. the ::-1)
    return np.ascontiguousarray(frames[:, :, ::-1, :])


def image_resize(image, l, inter=cv2.INTER_LINEAR):
    """
    Resize an image, preseving aspect ratio, so that its smaller edge is l

    Returns the resized image
    """

    # initialize the dimensions of the image to be resized and grab the image size.
    # we assume the image is in opencv and np's HWC format
    (h, w) = image.shape[:2]

    # operate with h if h <= w, else operate with w
    if h <= w:
        dim = (int(w * l / h), l)
    else:
        dim = (l, int(h * l / w))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def loop_list(frame_list, num_frames):
    """
    Fill a list by appending elements from it in a loop it until it has num_frames elements
    """

    original_list_length = len(frame_list)  # original length of the list
    list_counter = 0  # which element of the list we are on

    # append to list in a loop until we reach num_frames
    while len(frame_list) < num_frames:
        frame_list.append(frame_list[list_counter])
        list_counter += 1
        if list_counter >= original_list_length:
            list_counter = 0


def load_video(filename, resize_len, min_video_len):
    """
    Load a video, rescaling the shorter dimension to self.rescale_size, and returning as a DHWC numpy array, with
    pixels rescaled between 0 and 1. This returns the full video. If necessary, video frames are looped until the
    resulting array has self.min_video_len frames.
    """

    # Load video and open frame by frame
    cap = cv2.VideoCapture(filename)

    # initialize frame list
    frame_list = []

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # resize frame, preserving aspect ratio, so that its smaller edge is correctly sized
            resized = image_resize(frame, resize_len)

            # rescale pixes to [0, 1]
            resized = resized / 255

            # add frame to list
            frame_list.append(resized)
        else:
            # end of video
            break

    # release the video capture object
    cap.release()

    # if the number of frames we have is less than self.min_video_len, fill the list by looping the video
    if min_video_len is not None and len(frame_list) < min_video_len:
        loop_list(frame_list, min_video_len)

    # convert to np array
    frames = np.array(frame_list, dtype=np.float32)

    # returns in DHWC, np,array, np.float32 format
    return frames


class VideoDataset(torch.utils.data.Dataset):
    """
    A dataset which returns a sampled, cropped video clip as a tensor.

    The intention of this dataset is to permit both training modes and inference modes, In training modes, in general,
    the video is rescaled in an aspect ratio preserving way, randomly cropped, then L consecutive frames are randomly
    sampled. In test modes, the model often needs to be run on multiple uniform crops in both time and space. Instead of
    doing something like returning a large, stacked tensor with multiple crops in test mode, we instead provide the
    option to not crop (set crop_size <= 0 or to None) or sample frames (set frames_to_sample <= 0 or to None), return
    the entire video as a tensor, and leave further preprocessing to be done by the user at inference-time.

    In training then, crop_size should be provided, center_crop should be set to False, and frames_to_sample should also
    be provided. In inference mode, if additional preprocessing is required, cropping and frame sampling can be turned
    off by setting crop_size <= 0 or to None, and frames_to_sample <= 0 or to None. If a center-cropped video is all
    that is required for inference, that can be accomodated by setting center_crop to True.
    """
    def __init__(self, data_dict, class_dict, resize_len, crop_size, center_crop, min_video_len, frames_to_sample,
                 normalize_mean=(0.43216, 0.394666, 0.37645), normalize_std=(0.22803, 0.22145, 0.216989),
                 random_horizontal_flip=False):
        """
        Takes the following arguments:
        data_dict: (dict) a dictionary formatted like {filename: label_string}
        class_dict: (dict) a dictionary formatted like {label_string: class_index}
        resize_len: (int) the value to resize the video's smaller edge to, while preserving aspect ratio
        crop_size: (int) Size to randomly crop the video to. If <=0 or None, no cropping is done (for evaluation modes
        where cropping needs to happen in a specific way incompatible with batching)
        center_crop: (bool) Whether to take a center crop (True) or random crop (False). If crop_size <=0 or is None,
        this argument is ignored
        min_video_len: (int) If the full-length video is fewer than this many clips, loop the video from the beginning
        until this number of frames is met, before sampling frames. This is to handle very short videos. If you want
        this behavior turned off, set this value <= 0
        frames_to_sample: (int) Number of frames to return, randomly sampled from min_video_len or more frames. If this
        argument is <=0 or None, then all frames are returned. This is to support frame sampling modes for evaluation
        normalize_mean: (float tuple) After rescaling pixels from [0, 255] to [0, 1], use this and normalize_std to
        normalize pixels, assuming image is in RGB channel order. The defaults are from the UCF101 dataset.
        normalize_std: (float tuple) see normalize_mean
        random_horizontal_flip: (bool) Whether to randomly horizontally flip the video
        """

        # if we are cropping, make sure crop_size is smaller than the frame size
        if crop_size is not None and crop_size > 0:
            assert crop_size < resize_len, \
                'VideoDataset(): crop size ({}) must be smaller than resized dimension ({})'.format(crop_size,
                                                                                                    resize_len)

        # make sure normalization parameters have the same length
        if normalize_mean is not None and normalize_std is not None:
            assert len(normalize_mean) == len(normalize_std), \
                'VideoDataset(): normalize_mean and normalize_std have different lengths.'

        # do a check here that every file in the data_dict exists
        no_missing_files = True
        for filename in data_dict.keys():
            if not os.path.isfile(filename):
                print('Missing file: {}'.format(filename))
                no_missing_files = False
        assert no_missing_files, 'VideoDataset(): Some files in the provided dictionary could not be found.'

        self.data_list = list(data_dict.items())
        self.class_dict = class_dict
        self.resize_len = resize_len
        self.crop_size = crop_size
        self.center_crop = center_crop
        self.min_video_len = min_video_len
        self.frames_to_sample = frames_to_sample
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.random_horizontal_flip = random_horizontal_flip

    def __getitem__(self, idx):
        filename = self.data_list[idx][0]  # file path to video
        class_index = self.class_dict[self.data_list[idx][1]]  # label, as an integer

        # load video. This function also rescales to [0, 1] and loops video as needed to have at least
        # self.min_video_len frames. This returns RGB frames as a float32 np array in DHWC order
        frames = self._load_video(filename, self.resize_len, self.min_video_len)

        # crop the video according to the input parameters
        if self.crop_size is not None and self.crop_size > 0:
            if self.center_crop:
                frames = self._center_crop(frames, self.crop_size)
            else:
                frames = self._random_crop(frames, self.crop_size)

        # random sample the frames if needed
        if self.frames_to_sample is not None and self.frames_to_sample > 0:
            frames = self._random_sample(frames, self.frames_to_sample)

        # normalize the frames
        for i in range(len(self.normalize_mean)):
            frames[:, :, :, i] = (frames[:, :, :, i] - self.normalize_mean[i]) / self.normalize_std[i]

        # horizontal flip the frames if needed
        if self.random_horizontal_flip and random.random() < 0.5:
            frames = self._horizontal_flip(frames)

        # swap axes to pytorch CDHW format
        frames = np.transpose(frames, axes=(3, 0, 1, 2))

        # convert frames and label to tensors and return
        return torch.from_numpy(frames), torch.as_tensor(class_index, dtype=torch.int64)

    def __len__(self):
        return len(self.data_list)

    def _random_sample(self, frames, sample_length):
        return random_sample(frames, sample_length)

    def _random_crop(self, frames, crop_size):
        return random_crop(frames, crop_size)

    def _center_crop(self, frames, crop_size):
        return center_crop(frames, crop_size)

    def _horizontal_flip(self, frames):
        return horizontal_flip(frames)

    def _load_video(self, filename, resize_len, min_video_len):
        return load_video(filename, resize_len, min_video_len)


if __name__ == '__main__':
    dataset = VideoDataset({'/home/daveboat/Documents/course5-work/Video_relationship_labeling/datasets/cat.mp4': 'cat'},
                           {'cat': 0}, 128, 112, False, 32, 32)

    clip_tensor, label_tensor = dataset.__getitem__(0)

    print(clip_tensor.size())
    print(label_tensor)
