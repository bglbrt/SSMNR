#!/usr/bin/env python

# os libraries
import os

# numerical and computer vision libraries
import torchvision.transforms as transforms

class AUGMENTER():
    '''
    Images augmenter / processer.
    '''

    def __init__(self, input_size):
        '''
        Initialization function.
        '''
        self.input_size = input_size

    def get_normalize_transforms(self):
        '''
        '''

        # initialise transform
        normalize_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])

        # return transforms
        return normalize_transforms

    def get_train_transforms(self):
        '''
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
        '''

        # initialise train transforms
        validation_transforms = transforms.Compose([
                    transforms.RandomCrop(self.input_size),
                    ])

        # return transforms
        return validation_transforms

    def get_process_transforms(self):
        '''
        '''

        # initialise process transforms
        process_transforms = transforms.Compose([transforms.ToPILImage()])

        # return transforms
        return process_transforms

    def get_denoise_transforms(self, padding):
        '''
        '''

        # load padding variables
        pad_hl, pad_hr, pad_vt, pad_vb = padding

        # initialise denoise transforms
        denoise_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Pad(padding=(pad_hl, pad_vt, pad_hr, pad_vb), padding_mode='edge')
                ])

        # return transforms
        return denoise_transforms
