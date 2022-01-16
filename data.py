#!/usr/bin/env python

# os libraries
import os
import copy

# numerical and computer vision libraries
import torch
import numpy as np
from PIL import Image

# dependencies
from utils import *
from filters import *

# data loading class for training
class LOADER():
    '''
    Images loader.
    '''

    def __init__(self, data, phase, n_channels, batch_size, steps_per_epoch, input_size, method, window, ratio, noise_type, sigma):
        '''
        Initialization function.

        Arguments:
            data: str
                - path to folder containing training and validation data
            phase: str
                - training or validation phase
            n_channels: int
                - number of channels in images (3 for RGB; 1 for grayscale)
            batch_size: int
                - number of images per batch
            steps_per_epoch: int
                - number of batches in one epoch
            method: str
                -  blind-spot masking method
            input_size: int
                - size of patches
            window: int
                - window for blind-spot masking method in UPS
            ratio: float
                - ratio for number of blind-spot pixels in patch
            noise_type: str
                - noise type from Gaussian (G), Poisson (P) and Impulse (I)
            sigma: int
                - noise parameter for creating labels - depends on distribution
        '''

        # add from arguments
        self.data = data
        self.phase = phase
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.input_size = input_size
        self.method = method
        self.window = window
        self.ratio = ratio
        self.noise_type = noise_type
        self.sigma = sigma

        # set number of items in each batch for training
        if self.phase == 'train':
            self.iter = self.steps_per_epoch * self.batch_size

        # set number of items in each batch for validation
        elif self.phase == 'validation':
            self.iter = int(0.2 * self.steps_per_epoch * self.batch_size)

        # define AUGMENTER for data transforms
        self.augmenter = AUGMENTER(self.input_size)

        # define normalisation transforms
        self.normalize_transforms = self.augmenter.get_normalize_transforms()

        # define train and validation transforms
        if phase == 'train':
            self.transforms = self.augmenter.get_train_transforms()
        elif phase == 'validation':
            self.transforms = self.augmenter.get_validation_transforms()

        # get images extensions
        image_extensions = get_extensions()

        # define paths to images
        paths = [i for i in os.listdir(self.data) if i.lower().endswith(image_extensions)]
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
        image = Image.open(os.path.join(self.data, self.paths[i]))
        image.load()

        # convert images
        if self.n_channels == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')

        elif self.n_channels == 1:
            if image.mode != 'L':
                image = image.convert('L')

        # compute transforms
        image = self.transforms(image)

        # convert image to array
        image = np.array(image)

        # add random Gaussian noise to label
        if self.noise_type == 'G':
            noise = np.rint(np.random.normal(0, self.sigma, (self.input_size, self.input_size, self.n_channels)))
            label = image + noise

        # add random Poisson noise to label
        elif self.noise_type == 'P':
            noise = np.rint(np.random.poisson(self.sigma, (self.input_size, self.input_size, self.n_channels)))
            label = image + noise

        # add random Impulse noise to label
        elif self.noise_type == 'I':
            noise = np.rint(np.random.uniform(0, 255, (self.input_size, self.input_size, self.n_channels)))
            mask = np.random.binomial(1, self.sigma, (self.input_size, self.input_size, self.n_channels))
            label = image
            label[mask] = noise[mask]

        # raise error if noise type of noise not implemented
        else:
            raise NotImplementedError('Noise type not implemented: please use either G (Gaussian), P (Poisson) or I (Impulse)')

        # compute mask
        input, mask = self.get_mask(label)

        # normalize
        input = torch.from_numpy(input.transpose((2, 0, 1))).contiguous().div(255)
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).contiguous()
        label = torch.from_numpy(label.transpose((2, 0, 1))).contiguous().div(255)

        # normalize input and label
        input = self.normalize_transforms(input)
        label = self.normalize_transforms(label)

        # return data as floats or as int
        return input.float(), label.float(), mask

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
            masked_data: numpy.ndarray
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

            # Uniform Pixel Selection (UPS) method
            if self.method == 'UPS':
                for i, (x, y) in enumerate(masked_indices):
                    x_inf = max(0, x - self.window // 2)
                    x_sup = min(data.shape[0]-1, x + self.window // 2)
                    y_inf = max(0, y - self.window // 2)
                    y_sup = min(data.shape[1]-1, y + self.window // 2)
                    x_pos = np.random.randint(x_inf, high=x_sup+1)
                    y_pos = np.random.randint(y_inf, high=y_sup+1)
                    masked_data[x, y, c] = data[x_pos, y_pos, c]

            # Gaussian (G) method
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

    def __init__(self, n_channels, input_size):
        '''
        Initialization function.

        Arguments:
            n_channels: int
                - number of channels in images (3 for RGB; 1 for grayscale)
            input_size: int
                - size of patches
        '''

        # add from arguments
        self.n_channels = n_channels
        self.input_size = input_size

        # define AUGMENTER for transforms
        self.augmenter = AUGMENTER(self.input_size)

    def split_image(self, image, slide, chop):
        '''
        Function to split larger image for denoising function.

        Argumnts:
            image: PIL.Image.Image
                - image to split for denoising
            slide: int
                - sliding window over images (must be between 1 and input_size-4)
            chop: int
                - least amount of padding to add because of sliding window

        Returns:
            patches: torch.Tensor
                - splitted image into patches as torch Tensor
            pad_hl: int
                - left padding
            pad_hr: int
                - right padding
            pad_vt: int
                - top padding
            pad_vb: int
                - bottom padding
        '''

        # convert images
        if self.n_channels == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')

        elif self.n_channels == 1:
            if image.mode != 'L':
                image = image.convert('L')

        # compute image width and height, and define input_size
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

        # convert image to Tensor and apply transforms
        data = torch.from_numpy(data.transpose((2, 0, 1))).contiguous().div(255)
        data = denoise_transforms(data).float()

        # unfold tensor into patches
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

        # define array to populate output image
        output = np.zeros(data.shape, dtype=int)

        # denormalise image
        for c in range(self.n_channels):
            data[c, :, :] = (0.5 * data[c, :, :]) + 0.5
            output[c, :, :] = np.clip((data[c, :, :] * 255).cpu().detach().numpy(), a_min=0, a_max=255).astype('int')

        # convert image to uint8
        output = output.astype('uint8')

        # convert to numpy array
        output = np.transpose(output, (1,2,0))

        # convert to PIL image
        image = Image.fromarray(output, 'RGB')

        # return image
        return image
