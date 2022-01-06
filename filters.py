#!/usr/bin/env python

# os libraries
import os

# numerical and computer vision libraries
import torchvision.transforms as transforms

class AUGMENTER(input_size):
    '''
    Images augmenter / processer.
    '''

    def __init__(self):
        '''
        Initialization function.
        '''
        self.input_size = input_size

    def get_train_transforms(self):
        '''
        '''

        # initialise train transforms
        train_transforms = transforms.Compose([
                    transforms.RandomCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])

        # return transforms
        return train_transforms

    def get_validation_transforms(self):
        '''
        '''

        # initialise train transforms
        validation_transforms = transforms.Compose([
                    transforms.RandomCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

        # return transforms
        return validation_transforms

    def get_eval_transforms(self):
        '''
        '''

        # initialise train transforms
        evaluation_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

        # return transforms
        return evaluation_transforms

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
        pad_h1, pad_v1, pad_h2, pad_v2 = padding

        # initialise denoise transforms
        denoise_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Pad(padding=(pad_h1, pad_v1, pad_h2, pad_v2), padding_mode='edge')
                ])

        # return transforms
        return denoise_transforms
