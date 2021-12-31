import os
from PIL import Image
import numpy as np

image = Image.open(os.path.join('/Users/benjamingilbert/Desktop/', 'Im29_f2.jpg'))
image.load()
image = image.convert('RGB')
type(image)

width, height = image.size


data = np.asarray(image)

data.shape

type(data)


X = np.zeros((20, 20))

X[[5, 6, 7], [0, 8, 1]] = 1.0

X
