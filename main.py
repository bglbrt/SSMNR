#!/usr/bin/env python

# os libraries
import argparse

# dependencies
from utils import *
from train import *

# parser initialisation
parser = argparse.ArgumentParser(description='Self-Supervised Methods for Noise Removal')

# training settings
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="Folder where training and testing data is located (default: data).")
parser.add_argument('--model', type=str, default='N2V', metavar='M',
                    help='Name of model for noise removal (default: N2V).')
parser.add_argument('--input_size', type=int, default=64, metavar='IS',
                    help='Model patches input size (default: 64).')
parser.add_argument('--n_feat', type=int, default=96, metavar='NF',
                    help='Number of feature maps of the first convolutional layer (default: 96).')
parser.add_argument('--mode', type=str, default='train', metavar='MD',
                    help='Training, denoising or evaluation mode (default: train).')
parser.add_argument('--weights_init_method', type=str, default='kaiming', metavar='WI',
                    help='Weights initialization method (default: Kaiming normal).')
parser.add_argument('--images_path', type=str, default=None, metavar='IP',
                    help='Path to image or directory of images to denoise (default: None).')
parser.add_argument('--slide', type=int, default=24, metavar='SL',
                    help='Sliding window size for denoising (default: 24).')
parser.add_argument('--use_cuda', type=bool, default=True, metavar='UC',
                    help='Use of GPU (default: True).')
parser.add_argument('--from_pretrained', type=bool, default=False, metavar='FP',
                    help='Use pre-trained weights (default: False)')
parser.add_argument('--weights', type=str, default=None, metavar='W',
                    help='Path to weights to use for fine-tuning (default: None)')
parser.add_argument('--loss', type=str, default='L2', metavar='L',
                    help='Loss function for training (default: L2)')
parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                    help='Batch size for training data (default: 64)')
parser.add_argument('--steps_per_epoch', type=int, default=100, metavar='SE',
                    help='Number of steps per epoch for training (default: 200)')
parser.add_argument('--epochs', type=int, default=400, metavar='E',
                    help='Number of epochs to train the model (default: 400)')
parser.add_argument('--masking_method', type=str, default="UPS", metavar='MM',
                    help='Masking method (default: UPS)')
parser.add_argument('--window', type=int, default=7, metavar='WI',
                    help='Window for masking method (default: 20)')
parser.add_argument('--ratio', type=float, default=1/64, metavar='R',
                    help='Ratio for masking method (default: 1/64)')
parser.add_argument('--sigma', type=float, default=25, metavar='S',
                    help='Noise standard deviation for creating labels (default: 25)')
parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                    help='Weight decay for AdamW optimiser (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')

# main function
def main():
    '''
    Main function for training and evaluating models, and denoising single images.
    '''

    # parse arguments for training settings
    args = parser.parse_args()

    # define training script and model using arguments
    worker = TRAINER(args)

    # train mode
    if args.mode == 'train':
        worker.train(args.model)

    # eval mode
    elif args.mode == 'denoise':
        worker.denoise(args.model)

    # denoise mode (denoising a single image or images in a directory)
    elif args.mode == 'eval':
        worker.eval()

# run main function
if __name__ == '__main__':
    main()
