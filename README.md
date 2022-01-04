# SSMNR | RECVIS MVA Project 2021

Image denoising is the task of removing noise from an image, which can be formulated as the task of separating the noise signal from the meaningful information in images. Traditionally, this has been addressed both by spatial domain methods and transfer domain methods. However, from around 2016 onwards, image denoising techniques based on neural networks have started to outperfom these methods, with CNN-based denoisers obtaining impressive results.

One limitation to the use of neural-network based denoisers in many applications is the need for extensive, labeled datasets containing both noised images, and ground-truth, noiseless images. In answer to this, multiple works have explored the use of semi-supervised approaches for noise removal, requiring either noised image pairs but no clean target images (*Noise2Noise*) or, more recently, no additional data than the noised image (*Noise2Void*). This project aims at studying these approaches for the task of noise removal, re-implementing them in PyTorch.

This repository contains our code for this task. This code is heavily based on both the original implementation of the *Noise2Void* [article](https://arxiv.org/abs/1811.10980) available [here](https://github.com/juglab/n2v) and on other implementations and PyTorch/TensorFlow reproducibility challenges.

## Data

Data used to train and evaluate the algorithm consists in multiple datasets, including:
- [CBSD68](https://github.com/clausmichele/CBSD68-dataset)
- ...

No additional, noiseless, data was used to train the models, although the two architectures used for classification are used with pre-trained weights (trained on ImageNet).

## Usage

To reproduce these results, please start by cloning the repository locally:

```
git clone https://github.com/bglbrt/SSMNR.git
```

Then, install the required libraries:

```
pip install -r requirements.txt
```

#### Denoising (with provided weights)

To denoise an image or multiple images in a specified directory, run:

```
python main.py --mode denoise --images_path "path/to/image/or/dir" --from_pretrained True --weights "models/model.pth"
```

#### Training

To train weights for Noise2Void using data in ``data``, run:

```
python main.py data "data" --model "N2V" --mode train"
```

#### Training

The options available for the script are:

  * `--data`:
    Folder where training and testing data is located.
    - default: *data*
  * `--model`:
    Name of model for noise removal.
    - default: *N2V*
  * `--mode`:
    Training, denoising or evaluation mode.
    - default: *train*    
  * `--images_path`:
    Path to image or directory of images to denoise.
    - default: **
  * `--slide`:
    Sliding window size for denoising.
    - default: *48*
  * `--use_cuda`:
    Use of GPU.
    - default: *True*
  * `--from_pretrained`:
    Use pre-trained weights.
    - default: *False*
  * `--weights`:
    Path to weights to use for denoising.
    - default: *None*
  * `--batch_size`:
    Batch size for training data.
    - default: *12*    
  * `--epochs`:
    Number of epochs to train the model.
    - default: *100*
  * `--masking_method`:
    asking method.
    - default: *UPS*
  * `--window`:
    indow for masking method.
    - default: *20*
  * `--ratio`:
    Ratio for masking method.
    - default: *0.05*
  * `--sigma`:
    Noise standard deviation for creating labels.
    - default: *1*
  * `--lr`:
    Learning rate.
    - default: *1e-4*
  * `--wd`:
    eight decay for AdamW optimiser.
    - default: *1e-4*    
  * `--seed`:
    Random seed.
    - default: *1*

## Required libraries

The files present on this repository require the following libraries (also listed in requirements.txt):
 - [NumPy](https://numpy.org)
 - [torch](https://pytorch.org)
 - [torchvision](https://pytorch.org/vision/stable/index.html)
 - [matplotlib](https://matplotlib.org)
