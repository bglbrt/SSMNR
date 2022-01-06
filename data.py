#!/usr/bin/env python

# os libraries
import os
import copy

# numerical and computer vision libraries
import torch
import numpy as np
from PIL import Image

# dependencies
from filters import *

# data loading class for training
class LOADER():
    '''
    Images loader.
    '''

    def __init__(self, data, method, input_size, window, ratio, sigma):
        '''
        Initialization function.

        Arguments:
            args: parser.args
                - parser arguments (from main.py)
        '''

        # add from arguments
        self.data_path = data
        self.method = method
        self.input_size = input_size
        self.window = window
        self.ratio = ratio
        self.sigma = sigma

        # add data transforms
        self.augmenter = AUGMENTER(self.input_size)
        self.train_transforms = self.augmenter.get_train_transforms()

        # filter images and sort
        image_extensions = ('.png')
        paths = [i for i in os.listdir(self.data_path) if i.lower().endswith(image_extensions)]
        paths.sort(key=lambda x: os.path.splitext(x)[0])
        self.paths = paths

    def __getitem__(self, i):
        '''
        Item getter.

        Arguments:
            i: int
                - index of path to image in self.paths

        Returns:
            data: numpy.ndarray
                - image as array
        '''

        # load image
        image = Image.open(os.path.join(self.data_path, self.paths[i]))
        image.load()

        # convert grayscale images to RGB (three-channel)
        if image.mode == 'L':
            image.convert('RGB')

        # compute width and height
        width, height = image.size

        # compute transforms
        image = self.train_transforms(image)

        # convert to array
        image_as_array = np.asarray(image)

        # compute label
        label = image_as_array + np.random.normal(0, self.sigma, (3, self.input_size, self.input_size))

        # compute mask
        input, mask = self.get_mask(copy.deepcopy(label))

        # return data
        return input, label, mask

    def __len__(self):
        '''
        Function to compute the number of images in dataset.

        Returns:
            len: int
                - number of images in dataset
        '''

        # return number of images
        return len(self.paths)

    def get_mask(self, data):
        '''
        Function to mask center pixel in image.

        Arguments:
            data: numpy.ndarray
                - input image as array (to be masked)

        Returns:
            data: numpy.ndarray
                - masked image as array
            mask: np.array
                - mask
        '''

        # get number of pixels to mask
        n_masked_pixels = int(self.input_size * self.input_size * self.ratio)

        # initialise masked data
        masked_data = data

        # initialise mask
        mask = np.zeros(data.shape)

        # randomly select indices to mask
        masked_x_indices = np.random.choice(data.shape[1], n_masked_pixels, replace=True)
        masked_y_indices = np.random.choice(data.shape[2], n_masked_pixels, replace=True)
        masked_indices = [(x, y) for x, y in zip(masked_x_indices, masked_y_indices)]

        # create mask
        mask[:, masked_x_indices, masked_y_indices] = 1.0

        # mask pixels in every channel
        for c in range(data.shape[2]):

            # UPS method
            if self.method == 'UPS':
                # TODO
                pass

            elif self.method == 'G':
                mean = np.mean(data[:, :, c], axis=(0, 1))
                std = np.std(data[:, :, c], axis=(0, 1))
                random_noise = np.random.normal(mean, std, n_masked_pixels)
                for i, (x, y) in enumerate(masked_indices):
                    masked_data[x, y, c] += random_noise[i]

        # return mask
        return masked_data, mask

# data processing class for denoising
class PROCESSER():
    '''
    Images processer.
    '''

    def __init__(self, input_size):
        '''
        Initialization function.
        '''

        # add from arguments
        self.input_size = input_size

        # add data transforms
        self.augmenter = AUGMENTER(self.input_size)
        self.process_transforms = self.augmenter.get_process_transforms()

    def split_image(self, image, slide):
        '''
        Function to split larger image for denoising function.

        Argumnts:
            image: PIL.Image.Image
                - image to split for denoising
            slide: int
                - sliding window over images (must be between 1 and input_size-4)

        Returns:
            patches: torch.Tensor
                - splitted image into patches as torch Tensor
            pad_v1: int
                - right vertical padding
            pad_v2: int
                - left vertical padding
            pad_h1: int
                - right horizontal padding
            pad_h2: int
                - left horizontal padding
        '''

        # convert grayscale images to RGB (three-channel)
        if image.mode == 'L':
            image.convert('RGB')

        # compute image width and height
        width, height = image.size

        # convert image to np.array
        data = np.asarray(image)

        # compute padding
        pad_h, pad_v = max(4, self.input_size - (width % self.input_size)), max(4, self.input_size - (height % self.input_size))
        pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
        pad_v1, pad_v2 = pad_v // 2, pad_v - pad_v // 2

        # define transforms
        denoise_transforms = self.augmenter.get_denoise_transforms((pad_h1, pad_v1, pad_h2, pad_v2))

        # apply transforms
        data = denoise_transforms(data)

        # unfold tensor
        patches = data.unfold(1, self.input_size, slide).unfold(2, self.input_size, slide)

        # return padding and patches
        return pad_v1, pad_v2, pad_h1, pad_h2, patches

    def process_image(self, data):
        '''
        Function to split larger image for denoising function.

        Argumnts:
            data: np.array
                - array containing raw denoised image

        Returns:
            image: PIL.Image.Image
                - denoised and processed image
        '''

        # denormalise image
        for c in range(3):
            data[c, :, :] = ([0.229, 0.224, 0.225][c] * data[c, :, :]) + [0.485, 0.456, 0.406][c]

        # define transforms
        process_transforms = transforms.Compose([transforms.ToPILImage()])

        # convert torch Tensor to PIL image
        image = self.process_transforms(data)

        # return image
        return image
