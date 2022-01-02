#!/usr/bin/env python

# os libraries
import os
import copy
import time

# numerical and computer vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# visualisation libraries
import matplotlib.pyplot as plt

# dependencies
from data import *
from utils import *
from models import *

class TRAINER():
    '''
    Training and evaluation methods.
    '''

    def __init__(self, args):
        '''
        Initialization function.

        Arguments:
            args: parser.args
                - parser arguments (from main.py)
        '''

        # add from parser
        self.data = args.data
        self.model = args.model
        self.mode = args.mode
        self.use_cuda = args.use_cuda
        self.from_pretrained = args.from_pretrained
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.masking_method = args.masking_method
        self.window = args.window
        self.ratio = args.ratio
        self.sigma = args.sigma
        self.lr = args.lr
        self.wd = args.wd
        self.seed = args.seed

        # add from dependencies
        self.image_transforms = transforms

        # set device for GPU if GPU available
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

        # set device to CPU otherwise
        else:
            self.device = torch.device("cpu")

    def load_model(self, model):
        '''
        Model loading function.

        Returns:
            model: torchvision.model
                - model to train
        '''

        # load model
        model = initialize_model(model)

        # return model
        return model

    def load_weights(self, model):
        '''
        Weights loading function.

        Arguments:
            model: torchvision.model
                - model to load weights on

        Returns:
            model: torchvision.model
                - model with loaded weights
        '''

        # create directory to save model weights
        if not os.path.isdir('models/'):
            os.makedirs('models/')

        # load last weights if required
        if self.from_pretrained:
            weights = torch.load('models/model.pth')
            model.load_state_dict(weights)

        # return model
        return model

    def load_data(self):
        '''
        Data loading function.

        Returns:
            data_loaders: dict
                - data loaders for torchvision model
        '''

        # load train data
        train_data = LOADER(os.path.join(self.data, 'train'), self.masking_method, self.window, self.ratio, self.sigma, self.image_transforms)

        # load validation data
        validation_data = LOADER(os.path.join(self.data, 'validation'), self.masking_method ,self.window, self.ratio, self.sigma, self.image_transforms)

        # define training data loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # define validation data loader
        validation_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # load dataloaders in dictionary
        data_loaders = {'train':train_loader, 'validation':validation_loader}

        return data_loaders

    def train(self, model):
        '''
        Training function.

        Arguments:
            model: torchvision.model
                - model to train

        Returns:
            model: torchvision.model
                - trained model
            losses: dict
                - training and validation losses (per epoch)
        '''

        print('#'*30)
        print('Initialising training \n')

        # set seed
        torch.manual_seed(self.seed)

        # create directory to save plots if necessary
        if not os.path.isdir('exports/'):
            os.makedirs('exports/')

        # training start time
        time_start = time.time()

        # load model
        model = self.load_model(self.model)

        # load weights
        model = self.load_weights(model)

        # set loss function
        loss_function = nn.MSELoss()

        # set optimizer
        optimizer = optim.RAdam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        # load data
        data_loaders = self.load_data()

        # intialise lists to store validation and train accuracies and losses
        losses = {'train':[], 'validation':[]}

        # initialise weights
        current_weights = copy.deepcopy(model.state_dict())

        print('#'*30)
        print('Starting training\n')

        # iterate over epochs
        for epoch in range(self.epochs):

            # epoch start time
            time_start_epoch = time.time()

            # print current epoch number
            print(('#' * (30)))
            print('Starting epoch {}/{}'.format(epoch+1, self.epochs))
            print(('-' * (30)))

            if (epoch > 0) and (epoch % 50 == 0):
                model_file = 'models/model_' + str(epoch) +'.pth'
                torch.save(current_weights, model_file)

            # iterate over train and validation phases
            for phase in ('train', 'validation'):

                # set model to training mode
                if phase == 'train':
                    model.train()

                # set model to evaluation mode for validation
                else:
                    model.eval()

                # initialise running loss
                running_loss = 0.0

                # iterate over data in batch
                for inputs, labels, masks in data_loaders[phase]:

                    # put data on GPU if available
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    masks = masks.to(self.device)

                    print(inputs.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):

                        # compute outputs
                        outputs = model(inputs)

                        # compute loss function
                        loss = loss_function(outputs, labels)

                        # backward
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # compute running loss
                    running_loss += loss.item() * inputs.size(0)

                # divide running loss by number of items in dataloader
                running_loss = running_loss / len(data_loaders[phase].dataset)

                # print epoch's loss and accuracy
                if phase == "train":
                    print('Training phase | Loss: {:.5f}'.format(running_loss))

                    # add loss to loss history
                    losses['train'].append(running_loss)

                elif phase == 'validation':
                    print('Validat. phase | Loss: {:.5f}'.format(running_loss))

                    # update current weights
                    current_weights = copy.deepcopy(model.state_dict())

                    # add loss to loss history
                    losses['validation'].append(running_loss)

                    # plot losses
                    plt.figure(figsize=(8,6))
                    plt.plot(range(len(losses['train'])), losses['train'], label = 'Training Loss', color='black', linestyle='dashed')
                    plt.plot(range(len(losses['validation'])), losses['validation'], label = 'Validation Loss', color='black')
                    plt.legend()
                    plt.xlabel('Number of epochs')
                    plt.ylabel('Loss')
                    plt.savefig('exports/loss.png')
                    plt.close()

            # print time since start of epoch
            time_end_epoch = time.time()
            time_epoch = time_end_epoch - time_start_epoch
            print(('-' * (30)))
            print('Epoch complete in {:.0f}m {:.0f}s \n'.format(time_epoch // 60, time_epoch % 60))

        # print time since start of epoch
        time_end = time.time()
        time_training = time_end - time_start
        print(('#' * (30)))
        print('Training complete in {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))

        # load best model weights
        model.load_state_dict(current_weights)

        # save last model
        model_file = 'models/model.pth'
        torch.save(current_weights, model_file)

        # print location of model weights
        print('Saved model to: ' + model_file)

        # return model and losses
        return model, losses

    def eval(self):
        '''
        '''

        if not self.from_pretrained:
            'Weights not provided!'

        # set seed
        torch.manual_seed(self.seed)

    def denoise(self, model, image_path):
        '''
        Single image denoising function.

        Arguments:
            model:
        '''

        print('#'*30)
        print('Initialising... \n')

        # set seed
        torch.manual_seed(self.seed)

        # create directory to save plots if necessary
        if not os.path.isdir('denoised/'):
            os.makedirs('denoised/')

        # training start time
        time_start = time.time()

        # load model
        model = self.load_model(self.model)

        # load weights
        model = self.load_weights(model)

        # set model to eval function
        model.eval()

        image = Image.open(image_path)
        image.load()

        # convert grayscale images to RGB (three-channel)
        if image.mode == 'L':
            image.convert('RGB')

        width, height = image.size
        pad_h = 64 - (width % 64)
        pad_v = 64 - (height % 64)

        pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
        pad_v1, pad_v2 = pad_v // 2, pad_v - pad_v // 2

        data = np.asarray(image)

        self.denoise_transforms = {'evaluation_in' : transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.Pad(padding=(pad_h1, pad_v1, pad_h2, pad_v2), padding_mode='edge')
                                ]),
                                'evaluation_out' : transforms.Compose([
                                    transforms.ToPILImage()
                                ])}

        data = self.denoise_transforms['evaluation_in'](data)

        slide = 32

        patches = data.unfold(1, 64, slide).unfold(2, 64, slide)

        denoised_patches = torch.zeros_like(patches)

        new_data = torch.zeros_like(data)
        div = torch.zeros_like(data)

        for i in range(patches.shape[1]):
            for j in range(patches.shape[2]):
                patch = patches[:, i, j, :, :]
                denoised_patch = model(torch.unsqueeze(patch, 0)).squeeze()

                new_data[:, slide*i + 2:(slide*i)+64 -2, slide*j + 2:(slide*j)+64 - 2] += denoised_patch[:, 2:-2, 2:-2]
                div[:, slide*i + 2:(slide*i)+64 -2, slide*j + 2:(slide*j)+64 - 2] += torch.ones_like(denoised_patch)[:, 2:-2, 2:-2]

        new_data = new_data / div

        #denoised_data = denoised_patches.reshape(3, height+pad_v, width+pad_h,)

        # TEMPORARY
        denoised_data = new_data[:, pad_v1:-pad_v2, pad_h1:-pad_h2]

        for c in range(3):
            denoised_data[c, :, :] = ([0.229, 0.224, 0.225][c] * denoised_data[c, :, :]) + [0.485, 0.456, 0.406][c]

        denoised_image = self.denoise_transforms['evaluation_out'](denoised_data)

        denoised_image.save(os.path.join('denoised', os.path.basename(image_path)))
