#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
import tensorflow as tf

import os
import re
import time
import h5py
import skimage
import numpy as np

from glob import glob
from PIL import Image


def load_dataset():
    hdf5_path = 'data/dataset.hdf5'
    hdf5_file = h5py.File(hdf5_path, 'r')
    
    X_train = hdf5_file['train_inpt'][:]
    y_train = hdf5_file['train_real'][:]
    X_test = hdf5_file['test_inpt'][:]
    y_test = hdf5_file['test_real'][:]
    
    return X_train, X_test, y_train, y_test

# Load dataset
train_source_images, train_target_images, test_source_images, test_target_images = load_dataset()

train_source_images = train_source_images.reshape(train_source_images.shape[0], 256, 256, 3).astype('float32')
train_source_images = train_source_images / 255.
train_target_images = train_target_images.reshape(train_target_images.shape[0], 256, 256, 3).astype('float32')
train_target_images = train_target_images / 255.
test_source_images = test_source_images.reshape(test_source_images.shape[0], 256, 256, 3).astype('float32')
test_source_images = test_source_images / 255.
test_target_images = test_target_images.reshape(test_target_images.shape[0], 256, 256, 3).astype('float32')
test_target_images = test_target_images / 255.

# Check dataset sizes
print(train_source_images.shape)
print(train_target_images.shape)
print(test_source_images.shape)
print(test_target_images.shape)

BUFFER_SIZE = 400  # train_source_images.shape[0] # number of training images
BATCH_SIZE = 4 # training batch size (memory dependent)
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Batch the data. I've eliminated shuffling here.
train_source_dataset = tf.data.Dataset.from_tensor_slices(train_source_images).batch(BATCH_SIZE)
train_target_dataset = tf.data.Dataset.from_tensor_slices(train_target_images).batch(BATCH_SIZE)

# Generator
#
# The Image-to-Image paper notes that it uses the U-Net as the generator.
def Generator(input_shape):
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.Input(input_shape)
    x = inputs

    # Encoder network
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv0')(x)
    conv0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv1')(conv0)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv2')(conv1)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv3')(conv2)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv4')(conv3)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv5')(conv4)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv6')(conv5)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='enc_conv7')(conv6)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    conv7 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Fully-connected layers to allow parts to move around.
    # If you are running out of memory, you can comment some of them out.

    # Flatten -> Dense -> LeakyReLU -> Dense -> LeakyReLU -> Dense -> LeakyReLU -> Dense -> LeakyReLU -> Dense -> LeakyReLU -> Reshape
    x = tf.keras.layers.Flatten()(conv7)
    x = tf.keras.layers.Dense(512, input_shape=(1, 1, 512), name='dense1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Dense(512, input_shape=(1, 1, 512), name='dense2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Dense(512, input_shape=(1, 1, 512), name='dense3')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Dense(512, input_shape=(1, 1, 512), name='dense4')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Dense(512, input_shape=(1, 1, 512), name='dense5')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Reshape((1, 1, 512))(x)

    # Decoder network
    # tf.keras.layers.Conv2DTrans -> BatchNorm -> Dropout -> ReLU
    x = tf.keras.layers.Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv7')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> Dropout -> ReLU
    x = tf.keras.layers.Concatenate(axis=-1, name='concat6')([conv6, x])
    x = tf.keras.layers.Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv6')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> Dropout -> ReLU
    x = tf.keras.layers.Concatenate(axis=-1, name='concat5')([conv5, x])
    x = tf.keras.layers.Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv5')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> ReLU
    x = tf.keras.layers.Concatenate(axis=-1, name='concat4')([conv4, x])
    x = tf.keras.layers.Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv4')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> ReLU
    x = tf.keras.layers.Concatenate(axis=-1, name='concat3')([conv3, x])
    x = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv3')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> ReLU
    x = tf.keras.layers.Concatenate(axis=-1, name='concat2')([conv2, x])
    x = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> ReLU
    x = tf.keras.layers.Concatenate(axis=-1, name='concat1')([conv1, x])
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False, name='dec_conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    # Concat -> tf.keras.layers.Conv2DTrans -> TanH
    x = tf.keras.layers.Concatenate(axis=-1, name='concat0')([conv0, x])
    outputs = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same',
                                              kernel_initializer=initializer,
                                              use_bias=False, activation='tanh', 
                                              name='dec_conv0')(x)

    # Return model. 
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Generator')


generator = Generator((256, 256, 3))
generator.summary()


# Discriminator
# 
# The discriminator takes the input shape of the image file and in our case it's (256, 256, 3).
def Discriminator(source_shape, target_shape):
    initializer = tf.random_normal_initializer(0., 0.02)

    input_image = tf.keras.Input(source_shape, name='input_image')
    target_image = tf.keras.Input(target_shape, name='target_image')

    x = tf.keras.layers.Concatenate(axis=-1, name='concat')([input_image, target_image])

    # Conv -> LeakyReLU
    x = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='valid', use_bias=False,
                               kernel_initializer=initializer, name='disc_conv0')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='valid', use_bias=False,
                               kernel_initializer=initializer, name='disc_conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='valid', use_bias=False,
                               kernel_initializer=initializer, name='disc_conv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Conv -> BatchNorm -> LeakyReLU
    x = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), name='disc_conv3', 
                               kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Conv -> Sigmoid
    x = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_last')(x)
    outputs = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), name='validity', use_bias=False,
                                     kernel_initializer=initializer, activation='sigmoid')(x)

    # Return model
    return tf.keras.Model(inputs=[input_image, target_image], outputs=outputs, name='Discriminator')


discriminator = Discriminator((256, 256, 3),(256,256,3))
discriminator.summary()


# Define Loss functions for Generator and Discriminator
# l1_weight and gan_weight are taken from the Image-to-Iamge paper.
# The numbers scale the two components of the loss function in the GAN.
l1_weight = 100.0
gan_weight = 1.0

# Epsilon
epsilon = 1e-12


def discriminator_loss(real_output, fake_output):
    total_loss = tf.reduce_mean(-(tf.math.log(real_output + epsilon) + tf.math.log(1 - fake_output + epsilon)))
    return total_loss


def generator_loss(fake_output, target_images, generated_images):
    gen_loss_GAN = tf.reduce_mean(-tf.math.log(fake_output + epsilon))
    gen_loss_L1 = tf.reduce_mean(tf.math.abs(target_images - generated_images))
    total_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
    return total_loss


# In[ ]:


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.99, epsilon=epsilon)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.99, epsilon=epsilon)


# ### Checkpoints

# In[ ]:


checkpoint_dir = 'data/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Training
EPOCHS = 500

# The use of `tf.function` causes the function to be "compiled".
@tf.function
def train_step(source_images, target_images, epoch_num):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(source_images, training=True)

        real_output = discriminator((source_images, target_images), training=True)
        fake_output = discriminator((source_images, generated_images), training=True)

        gen_loss = generator_loss(fake_output, target_images, generated_images)
        disc_loss = discriminator_loss(real_output, fake_output)

        report_gen_loss = tf.math.reduce_sum(gen_loss)
        report_disc_loss = tf.math.reduce_sum(disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return report_gen_loss, report_disc_loss


def train(source_dataset, target_dataset, epochs):
    epoch_start = 0

    # Set up history reporting
    history = {}
    gen_loss_list = []
    disc_loss_list = []

    for epoch in range(epochs):
        start = time.time()

        #source_iterator = source_dataset.make_one_shot_iterator()
        source_index = 0
        total_gen_loss = 0
        total_disc_loss = 0

        for image_target_batch in target_dataset:
            image_source_batch = train_source_images[np.array(list(range(source_index, source_index + BATCH_SIZE)))]
            report_gen_loss, report_disc_loss = train_step(image_source_batch, image_target_batch, epoch)
            source_index = source_index + BATCH_SIZE
            total_gen_loss = total_gen_loss + report_gen_loss
            total_disc_loss = total_disc_loss + report_disc_loss

        # Record and print the losses
        total_gen_loss_adj = (total_gen_loss / BUFFER_SIZE) / l1_weight
        total_disc_loss_adj = total_disc_loss / BUFFER_SIZE
        gen_loss_list.append(total_gen_loss_adj)
        disc_loss_list.append(total_disc_loss_adj)
        
        print("Generator Loss {}".format(total_gen_loss_adj))
        print("Discriminator Loss {}".format(total_disc_loss_adj))
        
        history['gen_loss'] = gen_loss_list
        history['disc_loss'] = disc_loss_list

        # Save images every 5 epochs.
        if (epoch + 1) % 5 == 0:
            generated_image = generator(test_source_images[np.array([0])], training=False)
            prediction = np.reshape(generated_image, (256, 256, 3))
            final_image = np.clip((prediction * 255), 0, 255).astype(np.uint8)
            generated_image2 = Image.fromarray(final_image)
            generated_image2.save('data/sample_output1_{:04d}.jpg'.format(epoch + 1))

            generated_image = generator(test_source_images[np.array([1])], training=False)
            prediction = np.reshape(generated_image, (256, 256, 3))
            final_image = np.clip((prediction * 255), 0, 255).astype(np.uint8)
            generated_image2 = Image.fromarray(final_image)
            generated_image2.save('data/sample_output2_{:04d}.jpg'.format(epoch + 1))

            generated_image = generator(test_source_images[np.array([2])], training=False)
            prediction = np.reshape(generated_image, (256, 256, 3))
            final_image = np.clip((prediction * 255), 0, 255).astype(np.uint8)
            generated_image2 = Image.fromarray(final_image)
            generated_image2.save('data/sample_output3_{:04d}.jpg'.format(epoch + 1))

            generated_image = generator(test_source_images[np.array([3])], training=False)
            prediction = np.reshape(generated_image, (256, 256, 3))
            final_image = np.clip((prediction * 255), 0, 255).astype(np.uint8)
            generated_image2 = Image.fromarray(final_image)
            generated_image2.save('data/sample_output4_{:04d}.jpg'.format(epoch + 1))

        # Save the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # Print time every 1 epochs
        if (epoch + 1) % 1 == 0:
            print ('Time for epoch {} is {} seconds\n'.format(epoch + 1, time.time()-start))

    return history


history = train(train_source_dataset, train_target_dataset, EPOCHS)


# Restore the latest checkpoint and test
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Create a sample image
for i in range(25):
    generated_image = generator(test_source_images[np.array([i])], training=False)
    prediction = np.reshape(generated_image, (256, 256, 3))
    final_image = np.clip((prediction * 255), 0, 255).astype(np.uint8)
    generated_image = Image.fromarray(final_image)
    generated_image.save('dataset/final.{:04d}.jpg'.format(i + 1))

#plt.axis('off')
#plt.imshow(generated_image)

def generate_images(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show();


for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar)
