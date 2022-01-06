#!/usr/bin/env python

# os libraries
import os
import copy
import time

# numerical and computer vision libraries
import torch
import torch.nn as nn
import torch.optim as optim

# visualisation libraries
import matplotlib.pyplot as plt

# dependencies
from data import *
from utils import *
from models import *

# training, evaluation and images denoising class
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
        self.input_size = args.input_size
        self.mode = args.mode
        self.images_path = args.images_path
        self.slide = args.slide
        self.use_cuda = args.use_cuda
        self.from_pretrained = args.from_pretrained
        self.weights = args.weights
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.masking_method = args.masking_method
        self.window = args.window
        self.ratio = args.ratio
        self.sigma = args.sigma
        self.lr = args.lr
        self.wd = args.wd
        self.seed = args.seed

        # set device for GPU if GPU available
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

        # set device to CPU otherwise
        else:
            self.device = torch.device("cpu")

    def load_model(self):
        '''
        Model loading function.

        Returns:
            model: torchvision.model
                - model to train
        '''

        # load model
        model = initialize_model(self.model)
        model = model.to(self.device)

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
            weights: str or None
                - path to weights (.pth)
        '''

        # create directory to save model weights
        if not os.path.isdir('models/'):
            os.makedirs('models/')

        # load last weights if required
        if self.from_pretrained:
            try:
                weights = torch.load(self.weights)
                model.load_state_dict(weights)
            except Exception as e:
                raise Exception("Error! Argument from_pretrained was set to True but no weights were provided.")

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
        train_data = LOADER(os.path.join(self.data, 'train'), self.masking_method, self.input_size, self.window, self.ratio, self.sigma)

        # load validation data
        validation_data = LOADER(os.path.join(self.data, 'validation'), self.masking_method, self.input_size, self.window, self.ratio, self.sigma)

        # define training data loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # define validation data loader
        validation_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # load dataloaders in dictionary
        data_loaders = {'train':train_loader, 'validation':validation_loader}

        # return data loaders
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
        model = self.load_model()

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
                model_file = 'models/model' + '_' + self.model + '_' + str(epoch) +'.pth'
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

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):

                        # compute outputs
                        outputs = model(inputs)

                        # compute loss function
                        loss = loss_function(outputs * masks, labels * masks)

                        im = (labels * (masks)).detach().numpy()
                        for i in range(self.batch_size):
                            imi = im[i, :, :, :].squeeze()
                            for c in range(3):
                                imi[c, :, :] = imi[c, :, :] * 255
                            imi = np.transpose(imi, (1, 2, 0))
                            print(imi.shape)
                            imi = Image.fromarray(imi.astype(np.uint8))
                            imi.save('denoised/lala'+str(i)+'.jpg')

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
        print('Training complete in {:.0f}m {:.0f}s'.format(time_training // 60, time_training % 60))

        # load best model weights
        model.load_state_dict(current_weights)

        # save last model
        model_file = 'models/model' + '_' + self.model + '.pth'
        torch.save(current_weights, model_file)

        # print location of model weights
        print('Saved model to: ' + model_file)

        print('Please run python main.py --mode denoise --images_path "path/to/image" --from_pretrained True --weights "'+str(model_file)+'" to denoise an image.')
        print('#'*30)

        # return model and losses
        return model, losses

    def denoise(self, model):
        '''
        Denoising function for single image or directory of images.

        Arguments:
            model: torchvision.model
                - model to train
        '''

        print('#'*30)
        print('Initialising...')

        # check from_pretrained is set to True
        if not self.from_pretrained:
            'Please set from_pretrained argument to True.'

        # check weights are provided
        if not self.weights:
            'Weights not provided! Please provide path to weights (*.pth) in weights argument.'

        # set seed
        torch.manual_seed(self.seed)

        # create directory to save plots if necessary
        if not os.path.isdir('denoised/'):
            os.makedirs('denoised/')

        # training start time
        time_start = time.time()

        # initialise image processor
        processer = PROCESSER(self.input_size)

        # load model
        model = self.load_model()

        # load weights
        model = self.load_weights(model)

        # set model to eval function
        model.eval()

        # check if self.images_path contains a single image or a directory of images
        if os.path.isfile(self.images_path):
            images_path = [self.images_path]

        # check if self.images_path contains a single image or a directory of images
        elif os.path.isdir(self.images_path):
            images_path = [os.path.join(self.images_path, path) for path in os.listdir(self.images_path) if os.path.splitext(path)[-1] in ['.jpg', '.png']]

        # initialise slide variable
        slide = self.slide

        print('Number of images found to process: %i \n' % len(images_path))
        print('#'*30)
        print('Beginning denoising... \n')

        # iterate over images
        for image_path in images_path:

            print('-'*30)
            print('Starting image: %s' % image_path)

            # load image
            image = Image.open(image_path)
            image.load()

            # compute image width and height
            width, height = image.size

            # split image into patches
            pad_v1, pad_v2, pad_h1, pad_h2, patches = processer.split_image(image, slide)

            # initialise new image array and sliding window counter
            data = torch.zeros((3, height+pad_v1+pad_v1, width+pad_h1+pad_h1))
            div = torch.zeros((3, height+pad_v1+pad_v1, width+pad_h1+pad_h1))

            # put output image on device
            data = data.to(self.device)
            div = div.to(self.device)

            # compute mode output over all patches
            for i in range(patches.shape[1]):
                for j in range(patches.shape[2]):

                    # compute denoised patch
                    input = torch.unsqueeze(patches[:, i, j, :, :], 0).to(self.device)
                    denoised_patch = model(input).squeeze()

                    # add denoised patch to data array and update div array
                    data[:, slide*i + 2:(slide*i)+self.input_size - 2, slide*j + 2:(slide*j)+self.input_size - 2] += denoised_patch[:, 2:-2, 2:-2]
                    div[:, slide*i + 2:(slide*i)+self.input_size -2, slide*j + 2:(slide*j)+self.input_size - 2] += torch.ones_like(denoised_patch)[:, 2:-2, 2:-2]

            # divide values in data array by number of sliding windows per pixel
            data = data / div

            # remove padding
            data = data[:, pad_v1:-pad_v2, pad_h1:-pad_h2]

            # process image
            denoised_image = processer.process_image(data)

            # save image
            output_path = os.path.join('denoised', os.path.basename(image_path))
            denoised_image.save(output_path)

            print('Image denoised and saved to: %s' % output_path)
            print('-'*30 + '\n')

        # print time since start of epoch
        time_end = time.time()
        time_denoising = time_end - time_start
        print(('#' * (30)))
        print('Denoising complete in {:.0f}m {:.0f}s'.format(time_denoising // 60, time_denoising % 60))

    def eval(self, model):
        '''
        Evaluation function.

        Arguments:
            model: torchvision.model
                - model to train

        Returns:
            ...
        '''

        if (not self.from_pretrained) or (not self.weights):
            'Weights not provided! \
            Please set from_pretrained argument to True and provide path to weights (*.pth) in weights argument.'

        # set seed
        torch.manual_seed(self.seed)
