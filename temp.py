#!/usr/bin/env python

import torchvision.transforms as transforms
import numpy as np

import torch
from PIL import Image
import copy

def get_mask(data):
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
    n_masked_pixels = int(256 * 256 * 0.8)

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

        method = 'UPS'
        # UPS method
        if method == 'UPS':
            for i, (x, y) in enumerate(masked_indices):
                x_inf = max(0, x - 7// 2)
                x_sup = min(data.shape[0]-1, x + 7 // 2)
                y_inf = max(0, y - 7 // 2)
                y_sup = min(data.shape[1]-1, y + 7 // 2)
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

# load image
image = Image.open('/Users/benjamingilbert/Desktop/Unknown-1 23.59.07.png')
image.load()

# convert grayscale images to RGB (three-channel)
if image.mode != 'RGB':
    image = image.convert('RGB')

train_transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

# compute transforms
image = train_transforms(image)

image

# convert image to array
image = np.array(image)

# add random noise to label
noise = np.rint(np.random.normal(0, 25, (256, 256, 3)))
label = image + noise

#noise = np.clip(noise, a_min=0, a_max=255).astype('uint8')
#label = np.clip(label, a_min=0, a_max=255).astype('uint8')

# compute mask
input, mask = get_mask(label)

# convert back to PIL image
#input = Image.fromarray(input, 'RGB')

#input

#label = Image.fromarray(label, 'RGB')

#label

input = torch.from_numpy(input).div(255).permute(2, 0, 1)
mask = torch.from_numpy(mask).div(255).permute(2, 0, 1)
label = torch.from_numpy(label).div(255).permute(2, 0, 1)

normalize_transforms = transforms.Compose([
            transforms.Normalize(mean = 0.5, std = 0.5)
        ])

# normalize
input = normalize_transforms(input)
label = normalize_transforms(label)

# convert mask to tensor
mask =  mask.permute(2, 0, 1)

output = np.zeros(label.shape, dtype=int)

# denormalise image
for c in range(3):
    label[c, :, :] = (0.5 * label[c, :, :]) + 0.5
    output[c, :, :] = np.clip((label[c, :, :] * 255).cpu().detach().numpy(), a_min=0, a_max=255).astype('int')

output = output.astype('uint8')

output.shape

# convert to numpu array
output = np.transpose(output, (1,2,0))

# convert to PIL image
image = Image.fromarray(output, 'RGB')

image
















output = np.zeros(input.shape, dtype=int)

temp = torch.zeros_like(input)

# denormalise image
for c in range(3):
    input[c, :, :] = ([0.229, 0.224, 0.225][c] * input[c, :, :]) + [0.485, 0.456, 0.406][c]
    output[c, :, :] = np.clip((input[c, :, :] * 255).cpu().detach().numpy(), a_min=0, a_max=255).astype('int')


output = output.astype('uint8')

output.shape


# convert to numpu array
output = np.transpose(output, (1,2,0))

# convert to PIL image
image = Image.fromarray(output, 'RGB')

image
