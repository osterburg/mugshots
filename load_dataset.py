#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf

import os
import re
import h5py
import shutil
import skimage
import imageio
import numpy as np

from glob import glob
#from tqdm import tqdm
#tqdm.pandas(desc="progress-bar")
#
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# don't print matching warnings
import warnings
warnings.filterwarnings('ignore') 

# ### Download dataset and unpack it
_URL = 'https://s3.amazonaws.com/nist-srd/SD18/sd18.zip'

path_to_zip = tf.keras.utils.get_file('sd18.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'sd18/')


# ### Organize all images
if not os.path.isdir('data'):
    os.mkdir('data')

filenames = glob(PATH + 'single/f1_p1/*/*.png')
for filename in tqdm(filenames):
    indx = filename.split('/')[-1].split('_')[0]
    # remove leading zeros from index
    indx = re.sub(r'(?<!\d)0+', '', indx)
    side = filename.split('/')[-1].split('_')[2].split('.')[0].lower()
    new_file = 'data/mugshot_{}.{}.png'.format(side, indx)
    shutil.copyfile(filename, new_file)

# Convert Grayscale to RGB 
# Resize to (256, 256)
filenames = glob('data/*.png')
for filename in tqdm(filenames):
    im = skimage.io.imread(filename)
    im = skimage.color.gray2rgb(im)
    im = skimage.transform.resize(im, (256, 256), anti_aliasing=True)
    im = skimage.util.img_as_ubyte(im)
    skimage.io.imsave(filename, im)

# Flip L to R
filenames = glob('data/mugshot_l.*.png')
for filename in tqdm(filenames):
    im = skimage.io.imread(filename)
    im = np.fliplr(im)
    skimage.io.imsave(filename, im)
    # rename file
    new_filename = filename.replace('_l', '_r')
    os.rename(filename, new_filename)

if not os.path.isdir('tmp'):
    os.mkdir('tmp')
    
if not os.path.isdir('data/test'):
    os.mkdir('data/test')

# train test split
frnt_files = sorted(glob('data/mugshot_f.*.png'))
side_files = sorted(glob('data/mugshot_r.*.png'))

mylist = list(zip(frnt_files, side_files))
for f in mylist[:int(len(mylist)*0.8)]:
    shutil.move(f[0], 'tmp')
    shutil.move(f[1], 'tmp')
    
for f in mylist[int(len(mylist)*0.8):]:
    shutil.move(f[0], 'data/test')
    shutil.move(f[1], 'data/test')


# ### Image-to-Image paper describes that it randomly jitter each image
# 1. resize image up
# 2. randomly crop back to org size
# 3. randomly flip horizontally
def load(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image)
    
    return tf.dtypes.cast(image, tf.float32)

@tf.function()
def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = tf.image.random_crop(image, size=[256, 256, 3])

    # randomly mirroring
    image = tf.image.random_flip_left_right(image)

    return image


# normalizing the images to [-1, 1]
def normalize(image):
    return (image / 127.5) - 1


if not os.path.isdir('data/train'):
    os.mkdir('data/train')

filenames = sorted(glob('tmp/mugshot_f.*.png'))
for filename in filenames:
    name = filename.split('/')[-1]
    image = load(filename)
    for i in range(4):
        image = random_jitter(image)
        image = normalize(image)
        nname = name.replace('_f', '_f' + str(i))
        imageio.imwrite(os.path.join('data/train/', nname), image)


filenames = sorted(glob('data/train/mugshot_f?.*.png'))
for k, filename in enumerate(filenames):
    shutil.move(filename, 'data/train/mugshot_front.' + str(k) + '.png') 

filenames = sorted(glob('tmp/mugshot_r.*.png'))
for filename in tqdm(filenames):
    for i in range(4):
        nname = filename.replace('_r', '_r' + str(i))
        shutil.copy2(filename, nname)

filenames = sorted(glob('tmp/mugshot_r?.*.png'))
for k, filename in enumerate(filenames):
    shutil.move(filename, 'data/train/mugshot_side.' + str(k) + '.png')

# Save the image dataset into a HDF5
hdf5_path = 'data/dataset.hdf5'

# train images
frnt_path = 'data/train/mugshot_front.*.png'
side_path = 'data/train/mugshot_side.*.png'

frnt = glob(frnt_path)
side = glob(side_path)

train_inpt = frnt[0:int(len(frnt))]
train_real = side[0:int(len(side))]

# test images
frnt_path = 'data/test/mugshot_f.*.png'
side_path = 'data/test/mugshot_r.*.png'

frnt = glob(frnt_path)
side = glob(side_path)

test_inpt = frnt[0:int(len(frnt))]
test_real = side[0:int(len(side))]

# Define an array for each of train and test set with the shape
# (number of data, image_height, image_width, image_depth)
train_shape = (len(train_inpt), 256, 256, 3)
test_shape = (len(test_inpt), 256, 256, 3)
    
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_inpt", train_shape, np.int8)
hdf5_file.create_dataset("train_real", train_shape, np.int8)
hdf5_file.create_dataset("test_inpt", test_shape, np.int8)
hdf5_file.create_dataset("test_real", test_shape, np.int8)

for i in range(len(train_inpt)):
    img = skimage.io.imread(train_inpt[i])
    hdf5_file["train_inpt"][i, ...] = img[None]
    img = skimage.io.imread(train_real[i])
    hdf5_file["train_real"][i, ...] = img[None]

for i in range(len(test_inpt)):
    img = skimage.io.imread(test_inpt[i])
    hdf5_file["test_inpt"][i, ...] = img[None]
    img = skimage.io.imread(test_real[i])
    hdf5_file["test_real"][i, ...] = img[None]

hdf5_file.close()

# Check if the data is saved properly in the HDF5 file
# open the hdf5 file
hdf5_path = 'data/dataset.hdf5'
hdf5_file = h5py.File(hdf5_path, 'r')

# Get total number of samples
num_data = hdf5_file['train_inpt'].shape[0]
print(num_data)

# Cleaning up behind me
# Removing all png images
files = glob('data/*.png')
for file in files:
    os.remove(file)
    
# remove tmp dir    
shutil.rmtree('tmp')

# Remove downloaded data
shutil.rmtree(PATH)
