import os
import re
import shutil
from glob import glob

import numpy as np
import skimage
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def prep_data():
    # Downloading dataset
    print("\nDownloading mugshot dataset...")
    _URL = 'https://s3.amazonaws.com/nist-srd/SD18/sd18.zip'
    path_to_zip = tf.keras.utils.get_file('sd18.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'sd18/')

    # Creating data directory
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Moving images from single/f1_p1 to our data dir
    print("\nMoving images ")
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
    print("\nConvert gray2rgb and resize images...")
    filenames = glob('data/*.png')
    for filename in tqdm(filenames):
        im = skimage.io.imread(filename)
        im = skimage.color.gray2rgb(im)
        im = skimage.transform.resize(im, (256, 256), anti_aliasing=True)
        im = skimage.util.img_as_ubyte(im)
        skimage.io.imsave(filename, im)

    # Flip L to R
    print("Flip left to right...")
    filenames = glob('data/mugshot_l.*.png')
    for filename in tqdm(filenames):
        im = skimage.io.imread(filename)
        im = np.fliplr(im)
        skimage.io.imsave(filename, im)
        # rename file
        new_filename = filename.replace('_l', '_r')
        os.rename(filename, new_filename)

    # Combine F and R image
    print("\nCombine front and right image...")
    filenames = sorted(glob('data/mugshot_f.*.png'))
    k = 1
    for filename in tqdm(filenames):
        images = [filename, filename.replace('_f', '_r')]
        images = [Image.open(image) for image in images]

        min_shape = sorted([(np.sum(image.size), image.size) for image in images])[0][1]
        imgs_comb = np.hstack((np.asarray(image.resize(min_shape)) for image in images))

        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save('data/mugshot_comp.{}.png'.format(k))
        k += 1

    # train test split
    print("\nSplit dataset into train and test...")
    filenames = glob('data/mugshot_comp.*.png')
    if not os.path.isdir('data/train'):
        os.mkdir('data/train')
    for filename in filenames[:int(len(filenames) * 0.8)]:
        shutil.move(filename, 'data/train')

    if not os.path.isdir('data/test'):
        os.mkdir('data/test')
    for filename in filenames[int(len(filenames) * 0.8):]:
        shutil.move(filename, 'data/test')

    # Removing all png images
    print("\nCleaning up...")
    files = glob('data/*.png')
    for file in files:
        os.remove(file)

    # Remove downloaded data
    shutil.rmtree(PATH)
    print("\nDONE\n")
