#!/usr/bin/env python

# os libraries
import os

# numerical and computer vision libraries
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]),
    'validation' : transforms.Compose([
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
    'evaluation' : transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                }

class LOADER():
    '''
    Images loader.
    '''

    def __init__(self, data, method, window, ratio, sigma, data_transforms):
        '''
        Initialization function.

        Arguments:
            args: parser.args
                - parser arguments (from main.py)
        '''

        # add from arguments
        self.data_path = data
        self.method = method
        self.window = window
        self.ratio = ratio
        self.sigma = sigma

        # add data transforms
        self.data_transforms = {
            'train': transforms.Compose([
                        transforms.RandomCrop(64),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ]),
            'validation' : transforms.Compose([
                        transforms.RandomCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                               }

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
        n_channels = 3

        # compute transforms
        image = self.data_transforms['train'](image)

        # convert to array
        image_as_array = np.asarray(image)

        # compute mask
        input, mask = self.get_mask(image_as_array)

        # compute label
        label = image_as_array + np.random.normal(0, self.sigma, (n_channels, 64, 64))

        # create dict with data
        data = {'input':input, 'label':label, 'mask':mask}

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
        n_masked_pixels = int(4096 * self.ratio)

        # initialise masked data
        masked_data = data

        # initialise mask
        mask = np.zeros(data.shape)

        # randomly select indices to mask
        masked_x_indices = np.random.choice(data.shape[0], n_masked_pixels, replace=True)
        masked_y_indices = np.random.choice(data.shape[1], n_masked_pixels, replace=True)
        masked_indices = [(x, y) for x, y in zip(masked_x_indices, masked_y_indices)]

        # create mask
        mask[masked_x_indices, masked_y_indices] = 1.0

        # mask pixels in every channel
        for c in range(data.shape[2]):

            # UPS method
            if self.method == 'UPS':
                # TODO
                pass

            elif self.method == 'G':
                random_noise = np.random.normal(0, 10, n_masked_pixels)
                for i, (x, y) in enumerate(masked_indices):
                    masked_data[x, y, c] += random_noise[i]

        # return mask
        return masked_data, mask
