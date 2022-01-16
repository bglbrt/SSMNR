#!/usr/bin/env python

# numerical and computer vision libraries
import numpy as np

def get_extensions():
    '''
    Image extensions.
    '''

    extensions = ('.jpg', '.jpeg', '.png')

    return extensions

def PSNR(ground_truth_luminance, denoised_luminance):
    '''
    Function to compute the PSNR or an image.

    Arguments:
        - ground_truth_luminance: np.array
            luminance of RGB or grayscale image
        - denoised_luminance: np.array
            luminance of RGB or grayscale image
    '''

    # compute MSE
    MSE = np.mean(np.square(ground_truth_luminance - denoised_luminance))

    # compute PSNR
    PSNR_image = 20 * np.log10(255) - 10 * np.log10(MSE)

    # return PSNR
    return PSNR_image
