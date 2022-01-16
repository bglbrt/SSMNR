#!/usr/bin/env python

# os libraries
import os
import copy
import time

# numerical and computer vision libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFilter

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
                - parser arguments (from main)
        '''

        # add from parser
        self.data = args.data
        self.mode = args.mode
        self.images_path = args.images_path
        self.model = args.model
        self.n_channels = args.n_channels
        self.input_size = args.input_size
        self.masking_method = args.masking_method
        self.window = args.window
        self.n_feat = args.n_feat
        self.noise_type = args.noise_type
        self.ratio = args.ratio
        self.from_pretrained = args.from_pretrained
        self.weights = args.weights
        self.weights_init_method = args.weights_init_method
        self.loss = args.loss
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.slide = args.slide
        self.sigma = args.sigma
        self.lr = args.lr
        self.wd = args.wd
        self.use_cuda = args.use_cuda
        self.seed = args.seed

        # set image extensions
        self.image_extensions = get_extensions()

        # set device for GPU if GPU available
        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                raise Exception('Tried device to GPU (use_cuda==True) but no GPU was found!')

        # set device to CPU if asked
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
        model = initialize_model(self.model, self.n_channels, self.n_feat)

        # check if model is evaluation model
        if isinstance(model, str):

            # return model
            return model

        # return model otherwise
        else:

            # place model on GPU
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
        '''

        # create directory to save model weights
        if not os.path.isdir('models/'):
            os.makedirs('models/')

        # load pre-trained weights for fine-tuning or initialise wiehgts
        if self.mode == 'train':

            # load weights if required
            if self.from_pretrained:
                try:
                    weights = torch.load(self.weights)
                    model.load_state_dict(weights)
                except Exception as e:
                    raise Exception("Error! Argument from_pretrained was set to True but no weights were found!")

            # initialise them otherwise
            else:

                # disable gradients
                with torch.no_grad():

                    # initialise weights with a gaussian distribution
                    if self.weights_init_method == 'normal':
                        for m in model.modules():
                            if isinstance(m, nn.Conv2d):
                                nn.init.normal_(m.weight.data)
                                m.bias.data.zero_()

                    # initialise weights following 'Understanding the difficulty of training deep feedforward neural networks'
                    elif self.weights_init_method == 'xavier':
                        for m in model.modules():
                            if isinstance(m, nn.Conv2d):
                                nn.init.xavier_normal_(m.weight.data)
                                m.bias.data.zero_()

                    elif self.weights_init_method == 'kaiming':
                        for m in model.modules():
                            if isinstance(m, nn.Conv2d):
                                nn.init.kaiming_normal_(m.weight.data)
                                m.bias.data.zero_()

        # load weights for denoising or evaluation
        elif self.mode in ('denoise', 'eval'):
            try:
                weights = torch.load(self.weights)
                model.load_state_dict(weights)
            except Exception as e:
                raise Exception("Error! Argument from_pretrained was set to True but no weights were found!")

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
        train_data = LOADER(os.path.join(self.data, 'train'), 'train', self.n_channels, self.batch_size, self.steps_per_epoch, self.input_size, self.masking_method, self.window, self.ratio, self.noise_type, self.sigma)

        # load validation data
        validation_data = LOADER(os.path.join(self.data, 'validation'), 'validation', self.n_channels, self.batch_size, self.steps_per_epoch, self.input_size, self.masking_method, self.window, self.ratio, self.noise_type, self.sigma)

        # define training data loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

        # define validation data loader
        validation_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

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

        # print function start
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

        # compute number of parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of parameters to learn: ' + str(params))

        # print model architecture
        print('Model architecture:')
        print(model)
        print(' '*30)

        # set loss function
        if self.loss == 'L1':
            loss_function = nn.L1Loss()
        elif self.loss == 'L2':
            loss_function = nn.MSELoss()
        elif self.loss == 'Huber':
            loss_function = nn.HuberLoss()

        # set optimizer
        optimizer = optim.RAdam(model.parameters(), lr=self.lr, weight_decay=self.wd, betas=[0.9, 0.999])

        # load data
        data_loaders = self.load_data()

        # intialise lists to store validation and train accuracies and losses
        losses = {'train':[], 'validation':[]}

        # initialise weights
        current_weights = copy.deepcopy(model.state_dict())

        # initialise past validation losses for LR scheduler
        past_validation_losses = [np.inf]*10

        # print training start
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

            # save weights every 50 epochs
            if (epoch > 0) and (epoch % 50 == 0):
                model_file = 'models/model' + '_' + self.model + '_' + self.noise_type + str(int(self.sigma)) + '_' + str(epoch) +'.pth'
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

                        # backward
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # compute running loss
                    running_loss += loss.item() * inputs.size(0)

                # divide running loss by number of items in dataloader
                running_loss = running_loss / len(data_loaders[phase].dataset)

                # print epoch's loss and accuracy at training phase
                if phase == "train":
                    print('Training phase | Loss: {:.5f}'.format(running_loss))

                    # add loss to loss history
                    losses['train'].append(running_loss)

                # print epoch's loss and accuracy at validation phase, and update learning rate if required
                elif phase == 'validation':
                    print('Validat. phase | Loss: {:.5f}'.format(running_loss))

                    # change learning rate
                    past_validation_losses.append(running_loss)
                    past_validation_losses.pop(0)

                    # check if validation loss is not decreasing anymore
                    if all(i <= past_validation_losses[-1] for i in past_validation_losses):

                        # update learning rate
                        self.lr = self.lr / 2

                        # update optimizer
                        optimizer = optim.RAdam(model.parameters(), lr=self.lr, weight_decay=self.wd, betas=[0.9, 0.999])

                        # printe learning rate update
                        print('*'*30)
                        print('Learning rate update to: {:.0e}'.format(self.lr))
                        print('*'*30)

                        # reset past validation losses
                        past_validation_losses = [np.inf]*10

                    # update current weights
                    current_weights = copy.deepcopy(model.state_dict())

                    # add loss to loss history
                    losses['validation'].append(running_loss)

                    # plot losses
                    plt.figure(figsize=(12,8))
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

        # load last model weights
        model.load_state_dict(current_weights)

        # save last model
        model_file = 'models/model' + '_' + self.model + '_' +  self.noise_type + str(int(self.sigma)) + '.pth'
        torch.save(current_weights, model_file)

        # print location of model weights
        print('Saved model to: ' + model_file)

        print('Please run python main.py --mode denoise --images_path "path/to/image" --weights "'+str(model_file)+'" to denoise an image.')
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

        # print function start
        print('#'*30)
        print('Initialising...')

        # set seed
        torch.manual_seed(self.seed)

        # training start time
        time_start = time.time()

        # initialise image processor
        processer = PROCESSER(self.n_channels, self.input_size)

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
            images_path = [os.path.join(self.images_path, path) for path in os.listdir(self.images_path) if os.path.splitext(path)[-1] in self.image_extensions]

        # initialise slide variable
        slide = self.slide

        # print number of images to process and denoising start
        print('Number of images found to process: %i \n' % len(images_path))
        print('#'*30)
        print('Beginning denoising... \n')

        # create directory to save denoised images if necessary
        if not os.path.isdir('denoised/'):
            os.makedirs('denoised/')

        # iterate over images
        for image_path in images_path:

            # print start of image denoising
            print('-'*30)
            print('Starting image: %s' % image_path)

            # load image
            image = Image.open(image_path)
            image.load()

            # compute image width and height
            width, height = image.size

            # compute chop (least amount of padding to add because of sliding window)
            chop = (self.input_size - slide) // 2

            # split image into patches
            pad_hl, pad_hr, pad_vt, pad_vb, patches = processer.split_image(image, slide, chop)

            # initialise new image array and sliding window counter
            data = torch.zeros((3, height+pad_vt+pad_vb, width+pad_hl+pad_hr))

            # put output data on device
            data = data.to(self.device)

            # compute model output over all patches
            for i in range(patches.shape[1]):
                for j in range(patches.shape[2]):

                    # compute denoised patch
                    input = torch.unsqueeze(patches[:, i, j, :, :], 0).to(self.device)
                    denoised_patch = model(input).squeeze()

                    # add denoised patch to data array
                    data[:, slide*i + chop:(slide*i)+self.input_size - chop, slide*j + chop:(slide*j)+self.input_size - chop] += denoised_patch[:, chop:-chop, chop:-chop]

            # remove padding
            data = data[:, pad_vt:-pad_vb, pad_hl:-pad_hr]

            # process image
            denoised_image = processer.process_image(data)

            # save image to denoised folder
            output_path = os.path.join('denoised', os.path.basename(image_path))
            denoised_image.save(output_path)

            # print path to denoised image
            print('Image denoised and saved to: %s' % output_path)
            print('-'*30 + '\n')

        # print time since start of denoising
        time_end = time.time()
        time_denoising = time_end - time_start
        print(('#' * (30)))
        print('Denoising complete in {:.0f}m {:.0f}s'.format(time_denoising // 60, time_denoising % 60))

    def eval(self, model):
        '''
        Evaluation function (for mean PSNR).

        Arguments:
            model: torchvision.model
                - model to train
        '''

        # set seed
        torch.manual_seed(self.seed)

        # print start of evaluation
        print('#'*30)
        print('Initialising...')

        # set seed
        torch.manual_seed(self.seed)

        # training start time
        time_start = time.time()

        # initialise image processor
        processer = PROCESSER(self.n_channels, self.input_size)

        # load model
        model = self.load_model()

        # load weights
        model = self.load_weights(model)

        # set model to eval function
        model.eval()

        # check if self.images_path contains a single image or a directory of images
        if os.path.isfile(self.images_path):
            images_path = [self.images_path]
            ground_truth_images_path = [os.path.join('original', self.images_path)]

        # check if self.images_path contains a single image or a directory of images
        elif os.path.isdir(self.images_path):
            images_path = [os.path.join(self.images_path, path) for path in os.listdir(self.images_path) if os.path.splitext(path)[-1] in self.image_extensions]
            ground_truth_images_path = [os.path.join(self.images_path, 'original', path) for path in os.listdir(self.images_path) if os.path.splitext(path)[-1] in self.image_extensions]

        # initialise slide variable
        slide = self.slide

        # create directory to save denoised images if necessary
        if not os.path.isdir(os.path.join(self.images_path, self.model)):
            os.makedirs(os.path.join(self.images_path, self.model))

        # print number of images to process
        print('Number of images found to process: %i \n' % len(images_path))
        print('#'*30)
        print('Beginning denoising... \n')

        # initialise list of PSNRs
        PSNRS = []

        # iterate over images
        for image_path, ground_truth_image_path in zip(images_path, ground_truth_images_path):

            # print start of evaluation
            print('-'*30)
            print('Starting image: %s' % image_path)

            # load image
            image = Image.open(image_path)
            image.load()

            # load ground truth image
            ground_truth_image = Image.open(ground_truth_image_path)
            ground_truth_image.load()

            # compute image width and height
            width, height = image.size

            # check if model is not N2V or N2VT
            if isinstance(model, str):

                # check if model is a MEAN filter
                if model == 'MEAN':

                    # apply mean filter
                    denoised_image = image.filter(ImageFilter.Kernel((5, 5), [1/25]*25))

                # check if model is MEDIAN filter
                elif model == 'MEDIAN':

                    # apply median filter
                    denoised_image = image.filter(ImageFilter.MedianFilter(5))

                elif model == 'BM3D':

                    # import only if needed
                    import bm3d

                    # convert image into array
                    image_as_array = np.asarray(image) / 255

                    # denoise image with BM3D
                    denoised_image = bm3d.bm3d_rgb(image_as_array, sigma_psd=self.sigma/255)

                    # clip image and convert to np.uint8
                    denoised_image = np.clip(denoised_image * 255, a_min=0, a_max=255).astype(int).astype(np.uint8)

                    # convert image back to PIL.Image
                    denoised_image = Image.fromarray(denoised_image, 'RGB')

                else:
                    raise NotImplementedError('Error! Evaluation method not implemented!')

            # denoise with N2V or N2VT otherwise
            else:

                # define chop for sliding
                chop = (self.input_size - slide) // 2

                # split image into patches
                pad_hl, pad_hr, pad_vt, pad_vb, patches = processer.split_image(image, slide, chop)

                # initialise new image array
                data = torch.zeros((3, height+pad_vt+pad_vb, width+pad_hl+pad_hr))

                # put output image on device
                data = data.to(self.device)

                # compute model output over all patches
                for i in range(patches.shape[1]):
                    for j in range(patches.shape[2]):

                        # compute denoised patch
                        input = torch.unsqueeze(patches[:, i, j, :, :], 0).to(self.device)
                        denoised_patch = model(input).squeeze()

                        # add denoised patch to data array and update div array
                        data[:, slide*i + chop:(slide*i)+self.input_size - chop, slide*j + chop:(slide*j)+self.input_size - chop] += denoised_patch[:, chop:-chop, chop:-chop]

                # remove padding
                data = data[:, pad_vt:-pad_vb, pad_hl:-pad_hr]

                # process image
                denoised_image= processer.process_image(data)

            # save image
            output_path = os.path.join(self.images_path, self.model, os.path.basename(image_path))
            denoised_image.save(output_path)

            # convert to YCbCr
            denoised_image = denoised_image.convert('YCbCr')
            ground_truth_image = ground_truth_image.convert('YCbCr')

            # isolate luminance
            denoised_luminance = np.asarray(denoised_image)[:, :, 0]
            ground_truth_luminance = np.asarray(ground_truth_image)[:, :, 0]

            # compute PSNR
            PSNR_image = PSNR(ground_truth_luminance, denoised_luminance)

            # append PSNR to list of all PSNRs
            PSNRS.append(PSNR_image)

            # print results for image
            print('PSNR value for individual image: {:.2f}'.format(PSNR_image))
            print('-'*30 + '\n')

        # compute mean PSNRS
        MEAN_PSNR = np.mean(PSNRS)

        # print time since start of epoch
        time_end = time.time()
        time_denoising = time_end - time_start
        print(('#' * (30)))
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_denoising // 60, time_denoising % 60))

        # print mean PSNR
        print(('*' * (30)))
        print('Mean PSNR for test data (' + os.path.splitext(self.images_path)[0] + '): {:.2f}'.format(MEAN_PSNR))
        print(('*' * (30)))
