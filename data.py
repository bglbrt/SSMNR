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

    def __init__(self, data, phase, batch_size, steps_per_epoch, method, input_size, window, ratio, sigma):
        '''
        Initialization function.

        Arguments:
            args: parser.args
                - parser arguments (from main.py)
        '''

        # add from arguments
        self.data_path = data
        self.phase = phase
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.method = method
        self.input_size = input_size
        self.window = window
        self.ratio = ratio
        self.sigma = sigma

        # set number of items in loader
        if self.phase == 'train':
            self.iter = self.steps_per_epoch * self.batch_size

        elif self.phase == 'validation':
            self.iter = int(0.2 * self.steps_per_epoch * self.batch_size)

        # add data transforms
        self.augmenter = AUGMENTER(self.input_size)
        self.normalize_transforms = self.augmenter.get_normalize_transforms()
        if phase == 'train':
            self.transforms = self.augmenter.get_train_transforms()
        elif phase == 'validation':
            self.transforms = self.augmenter.get_validation_transforms()

        # filter images and sort
        image_extensions = ('.png', '.jpg', '.jpeg')
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
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # compute transforms
        image = self.transforms(image)

        # convert image to array
        image = np.array(image)

        # add random noise to label
        noise = np.rint(np.random.normal(0, self.sigma, (self.input_size, self.input_size, 3))).astype(np.uint8)
        label = image + noise

        # compute mask
        input, mask = self.get_mask(label)

        # convert back to PIL image
        input = Image.fromarray(input, 'RGB')
        label = Image.fromarray(label, 'RGB')

        # normalize
        input = self.normalize_transforms(input)
        label = self.normalize_transforms(label)

        # convert mask to tensor
        mask =  torch.tensor(mask).permute(2, 0, 1)

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
        return np.min([self.iter, len(self.paths)])

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
        masked_data = copy.deepcopy(data)

        # initialise mask
        mask = np.zeros(data.shape)

        # randomly select indices to mask
        masked_x_indices = np.random.choice(data.shape[0], n_masked_pixels, replace=True)
        masked_y_indices = np.random.choice(data.shape[1], n_masked_pixels, replace=True)
        masked_indices = [(x, y) for x, y in zip(masked_x_indices, masked_y_indices)]

        # create mask
        mask[masked_x_indices, masked_y_indices, :] = 1.0

        # mask pixels in every channel
        for c in range(data.shape[2]):

            # UPS method
            if self.method == 'UPS':
                for i, (x, y) in enumerate(masked_indices):
                    x_inf = max(0, x - self.window // 2)
                    x_sup = min(data.shape[0]-1, x + self.window // 2)
                    y_inf = max(0, y - self.window // 2)
                    y_sup = min(data.shape[1]-1, y + self.window // 2)
                    x_pos = np.random.randint(x_inf, high=x_sup+1)
                    y_pos = np.random.randint(y_inf, high=y_sup+1)
                    masked_data[x, y, c] = data[x_pos, y_pos, c]

            # G method
            elif self.method == 'G':
                random_noise = np.random.normal(0, 10, n_masked_pixels)
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

    def split_image(self, image, slide, chop):
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
        input_size = self.input_size

        # convert image to np.array
        data = np.asarray(image)

        # compute padding
        pad_h = chop*2 + (slide - ((width + 2*chop - input_size) % slide))
        pad_v = chop*2 + (slide - ((height + 2*chop - input_size) % slide))
        pad_hl, pad_hr = pad_h // 2, pad_h - pad_h // 2
        pad_vt, pad_vb = pad_v // 2, pad_v - pad_v // 2

        # define transforms
        denoise_transforms = self.augmenter.get_denoise_transforms((pad_hl, pad_hr, pad_vt, pad_vb))

        # apply transforms
        data = denoise_transforms(data)

        # unfold tensor
        patches = data.unfold(1, self.input_size, slide).unfold(2, self.input_size, slide)

        # return padding and patches
        return pad_hl, pad_hr, pad_vt, pad_vb, patches

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

        # convert to numpu array
        data = np.clip((data * 255).cpu().detach().numpy(), a_min=0, a_max=255).astype('uint8')
        data = np.transpose(data, (1,2,0))

        # convert to PIL image
        image = Image.fromarray(data, 'RGB')

        # return image
        return image
