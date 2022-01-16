# SSMNR | Self-Supervised Methods for Noise Removal

Image denoising is the task of removing noise from an image, which can be formulated as the task of separating the noise signal from the meaningful information in images. Traditionally, this has been addressed both by spatial domain methods and transfer domain methods. However, from around 2016 onwards, image denoising techniques based on neural networks have started to outperfom these methods, with CNN-based denoisers obtaining impressive results.

One limitation to the use of neural-network based denoisers in many applications is the need for extensive, labeled datasets containing both noised images, and ground-truth, noiseless images. In answer to this, multiple works have explored the use of semi-supervised approaches for noise removal, requiring either noised image pairs but no clean target images ([*Noise2Noise*](https://arxiv.org/abs/1803.04189)) or, more recently, no additional data than the noised image ([*Noise2Void*](https://arxiv.org/abs/1811.10980)). This project aims at studying these approaches for the task of noise removal, and re-implementing them in PyTorch.

This repository contains our code for this task. This code is heavily based on both the original implementation of the *Noise2Void* [article](https://arxiv.org/abs/1811.10980) available [here](https://github.com/juglab/n2v), on other implementations and PyTorch/TensorFlow reproducibility challenges ([here](https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising) and [here](https://github.com/hanyoseob/pytorch-noise2void)), on the U-NET Transformer architecture available [here](https://github.com/HXLH50K/U-Net-Transformer/), as well as some base code from our teachers for a project on bird species recognition.

## Data

Data used to train and evaluate the algorithm consists mostly in:
- [CBSD68](https://github.com/clausmichele/CBSD68-dataset)
- [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/)

No noiseless data was used to train the models.

## Usage

To reproduce these results, please start by cloning the repository locally:

```
git clone https://github.com/bglbrt/SSMNR.git
```

Then, install the required libraries:

```
pip install -r requirements.txt
```

#### Denoising images (with provided, pre-trained weights)

To denoise an image or multiple images from a specified directory, run:

```
python main.py --mode denoise --model "model" --images_path "path/to/image/or/dir" --weights "path/to/model/weights"
```

Provided pre-trained weights are formatted as: "models/model_"+`model_name`+_+`noise_type`+`sigma`+".pth".

Available weights are:
- weights for the N2V model:
  - `models/model_N2V_G5.pth`
  - `models/model_N2V_G10.pth`
  - `models/model_N2V_G15.pth`
  - `models/model_N2V_G25.pth`
  - `models/model_N2V_G35.pth`
  - `models/model_N2V_G50.pth`
- weights for the N2VT (N2V with U-NET Transformer) model:
  - `models/model_N2V_G5.pth` (please contact us to obtain weights)
  - `models/model_N2V_G10.pth` (please contact us to obtain weights)
  - `models/model_N2V_G25.pth` (please contact us to obtain weights)

Options available for denoising are:

* `--mode`:
  Training (train), denoising (denoise) or evaluation (eval) mode
  - default: *train*
* `--images_path`:
  Path to image or directory of images to denoise.
  - default: *None*
* `--model`:
  Name of model for noise removal
  - default: *N2V*
* `--n_channels`:
  Number of channels in images - i.e. RGB or Grayscale images
  - default: *3*
* `--weights`:
  Path to weights to use for denoising, evaluation, or fine-tuning when training.
  - default: *None*
* `--slide`:
  Sliding window size for denoising and evaluation
  - default: *32*      
* `--use_cuda`:
  Use of GPU or CPU
  - default: *32*

#### Evaluation

To evaluate a model using a dataset in a specified directory, run:

```
python main.py --mode eval --model "model" --images_path "path/to/image/or/dir" --weights "path/to/model/weights"
```

Note that the data located at `path/to/image/or/dir` must include a folder named `original` with noiseless images.

Evaluation methods include:
- N2V (Noise2Void with trained weights)
- N2VT (Noise2VoidTransformer with trained weights)
- BM3D (Block-Matching and 3D Filtering)
- MEAN (5x5 mean filter)
- MEDIAN (5x5 median filter)

Provided pre-trained weights for N2V and N2VT are formatted as: "models/model_"+`model_name`+_+`noise_type`+`sigma`+".pth".

Available weights are:
- weights for the N2V model:
  - `models/model_N2V_G5.pth`
  - `models/model_N2V_G10.pth`
  - `models/model_N2V_G15.pth`
  - `models/model_N2V_G25.pth`
  - `models/model_N2V_G35.pth`
  - `models/model_N2V_G50.pth`
- weights for the N2VT (N2V with U-NET Transformer) model:
  - `models/model_N2V_G5.pth`
  - `models/model_N2V_G10.pth`
  - `models/model_N2V_G25.pth`

Options available for evaluation are:

* `--mode`:
  Training (train), denoising (denoise) or evaluation (eval) mode
  - default: *train*
* `--images_path`:
  Path to image or directory of images to evaluate.
  - default: *None*
* `--model`:
  Name of model for noise removal
  - default: *N2V*
* `--n_channels`:
  Number of channels in images - i.e. RGB or Grayscale images
  - default: *3*
* `--weights`:
  Path to weights to use for denoising, evaluation, or fine-tuning when training.
  - default: *None*
* `--slide`:
  Sliding window size for denoising and evaluation
  - default: *32*      
* `--use_cuda`:
  Use of GPU or CPU
  - default: *32*

#### Training

To train weights for the N2V and N2VT models using data located in the ``data`` folder, run:

```
python main.py data "data" --model "N2V" --mode train"
```

Note that the `data` folder must contain two folders named `train` and `validation`.

Options available for training are:

  * `--data`:
    Folder where training and testing data is located.
    - default: *data*
  * `--mode`:
    Training (train), denoising (denoise) or evaluation (eval) mode
    - default: *train*
  * `--model`:
    Name of model for noise removal.
    - default: *N2V*
  * `--n_channels`:
    Number of channels in images - i.e. RGB or Grayscale images
    - default: *3*
  * `--input_size`:
    Model patches input size
    - default: *64*
  * `--masking_method`:
    Blind-spot masking method
    - default: *UPS*
  * `--window`:
    Window for blind-spot masking method in UPS
    - default: *5*
  * `--n_feat`:
    Number of feature maps of the first convolutional layer
    - default: *96*
  * `--noise_type`:
    Noise type from Gaussian (G), Poisson (P) and Impulse (I)
    - default: *G*
  * `--ratio`:
    Ratio for number of blind-spot pixels in patch
    - default: *1/64*
  * `--from_pretrained`:
    Train model from pre-trained weights
    - default: *False*
  * `--weights`:
    Path to weights to use for denoising, evaluation, or fine-tuning when training
    - default: *None*
  * `--weights_init_method`:
    Weights initialization method
    - default: *kaiming*
  * `--loss`:
    Loss function for training
    - default: *L2*
  * `--batch_size`:
    Batch size for training data
    - default: *64*
  * `--epochs`:
    Number of epochs to train the model.
    - default: *300*
  * `--steps_per_epoch`:
    Number of steps per epoch for training
    - default: *100*
  * `--sigma`:
    Noise parameter for creating labels - depends on distribution
    - default: *25*
  * `--lr`:
    Learning rate
    - default: *4e-4*
  * `--wd`:
    Weight decay for RAdam optimiser
    - default: *1e-4*  
  * `--use_cuda`:
    Use of GPU or CPU
    - default: *32*  
  * `--seed`:
    Random seed
    - default: *1*

## Required libraries

The files present on this repository require the following libraries (also listed in requirements.txt):
 - [BM3D](https://webpages.tuni.fi/foi/GCF-BM3D/index.html)
 - [NumPy](https://numpy.org)
 - [torch](https://pytorch.org)
 - [torchvision](https://pytorch.org/vision/stable/index.html)
 - [matplotlib](https://matplotlib.org)
