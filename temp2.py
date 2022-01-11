import os
import shutil

from PIL import Image

files_train = [file for file in os.listdir('data/train/') if file.endswith('.JPEG')]
files_validation = [file for file in os.listdir('data/validation/') if file.endswith('.JPEG')]

for file in files_train:
    im = Image.open('data/train/'+file)
    width, height = im.size
    if width < 64 or height < 64:
        os.remove('data/train/'+file)

for file in files_validation:
    im = Image.open('data/validation/'+file)
    width, height = im.size
    if width < 64 or height < 64:
        os.remove('data/validation/'+file)
