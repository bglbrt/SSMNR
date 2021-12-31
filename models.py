#!/usr/bin/env python

# numerical and computer vision libraries
import timm
import torch
import torch.nn as nn
import torchvision.models as models

class N2V(nn.Module):
    '''
    UNet model.
    '''

    def __init__(self):
        '''
        Initialization function.

        Arguments:
            :
        '''

        #
        super(N2V, self).__init__()

        # ENCODER

        self.encode = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # DECODER

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # OUTPUT

        self.output = nn.Conv2d(48, 3, 1)

    def forward(self, x):
        '''
        Forward function.

        Arguments:
            x: torch.Tensor
                - UNET input

        Returns:
            x: torch.Tensor
                - UNTE output
        '''

        # ENCODER

        encoder = self.encode(x)

        # DECODER

        decoder = self.decode(encoder)

        # OUTPUT

        x = self.output(decoder)

        # return x
        return x.double()

# define function to initialise transformer model
def initialize_model(model_name):
    '''
    Arguments:
        model_name: str
            - name of model

    Returns:
        model: torchvision.model
            - model
    '''

    # initialise model
    if model_name == 'N2V':

        # initialise N2V
        model = N2V()

    return model
