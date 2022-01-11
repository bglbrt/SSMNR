#!/usr/bin/env python

import os
import shutil
import random

# creating train / val
root_dir = 'data'
os.makedirs(root_dir +'/train')
os.makedirs(root_dir +'/validation')

# shuffling files
files = [file for file in os.listdir(root_dir) if file.endswith('.JPEG')]
random.shuffle(files)

train_files = files[:int(len(files)*0.95)]

print(len(train_files))

val_files = files[int(len(files)*0.95):]

print(len(val_files))

# create lists of files to move
train_files = [name for name in train_files]
val_files = [name for name in val_files]

# move images
for file in train_files:
    shutil.move(root_dir + '/' + file, root_dir + '/' + 'train' + '/' + file)

for file in val_files:
    shutil.move(root_dir + '/' + file, root_dir + '/' + 'val' + '/' + file)
