#!/usr/bin/env python

# numerical and computer vision libraries
import torchvision.transforms as transforms

class AUGMENTER():
    '''
    Images augmenter / processer.
    '''

    def __init__(self, input_size):
        '''
        Initialization function.

        Arguments:
            input_size: int
                - size of patches
        '''

        # add from arguments
        self.input_size = input_size

    def get_normalize_transforms(self):
        '''
        Normalizer.
        '''

        # initialise transform
        normalize_transforms = transforms.Compose([
                    transforms.Normalize(mean = 0.5, std = 0.5)
                ])

        # return transforms
        return normalize_transforms

    def get_train_transforms(self):
        '''
        Train transforms.
        '''

        # initialise train transforms
        train_transforms = transforms.Compose([
                    transforms.RandomCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()
                ])

        # return transforms
        return train_transforms

    def get_validation_transforms(self):
        '''
        Validation transforms.
        '''

        # initialise train transforms
        validation_transforms = transforms.Compose([
                    transforms.RandomCrop(self.input_size),
                    ])

        # return transforms
        return validation_transforms

    def get_denoise_transforms(self, padding):
        '''
        Denoise transforms.

        Arguments:
            padding: tuple
                - (pad_hl, pad_hr, pad_vt, pad_vb) with:
                    • pad_hl: int
                        - left padding
                    • pad_hr: int
                        - right padding
                    • pad_vt: int
                        - top padding
                    • pad_vb: int
                        - bottom padding
        '''

        # load padding variables
        pad_hl, pad_hr, pad_vt, pad_vb = padding

        # initialise denoise transforms
        denoise_transforms = transforms.Compose([
                    transforms.Normalize(mean=0.5, std=0.5),
                    transforms.Pad(padding=(pad_hl, pad_vt, pad_hr, pad_vb), padding_mode='reflect')
                ])

        # return transforms
        return denoise_transforms
