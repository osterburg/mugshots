from __future__ import absolute_import, division, print_function, unicode_literals

import time

import matplotlib.pyplot as plt
import tensorflow as tf


class GAN:
    def __init__(self):
        self.train_dir = 'data/train/'
        self.test_dir = 'data/test/'
        # l1_weight and gan_weight are taken from the Image-to-Iamge paper.
        # The numbers scale the two components of the loss function in the GAN.
        self.l1_weight = 100.0
        self.gan_weight = 1.0
        self.epsilon = 1e-12
        self.X = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 3])
        self.y = tf.compat.v1.placeholder(tf.int16, [None, 256, 256, 3])
        self.training = tf.compat.v1.placeholder(tf.bool)

    def Generator(self, input_shape):
        """
        Generator using an U-Net architecture as describt in the Image-to-Image paper
        including five fully connected Dense layers
        Args:
            input_shape:

        Return:
            tensorflow model
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        inputs = tf.keras.Input(input_shape)
        x = inputs

        # Encoder network
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv0')(x)
        conv0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv1')(conv0)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv2')(conv1)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv3')(conv2)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv4')(conv3)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv5')(conv4)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                   use_bias=False, name='enc_conv6')(conv5)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        conv6 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # tf.keras.layers.Conv -> BatchNorm -> LeakyReLU
        x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
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
                                            use_bias=False, name='dec_conv7')(conv7)
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
        x = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                            use_bias=False, name='dec_conv3')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.ReLU()(x)

        # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> ReLU
        x = tf.keras.layers.Concatenate(axis=-1, name='concat2')([conv2, x])
        x = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                            use_bias=False, name='dec_conv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.ReLU()(x)

        # Concat -> tf.keras.layers.Conv2DTrans -> BatchNorm -> ReLU
        x = tf.keras.layers.Concatenate(axis=-1, name='concat1')([conv1, x])
        x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer,
                                            use_bias=False, name='dec_conv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.ReLU()(x)

        # Concat -> tf.keras.layers.Conv2DTrans -> TanH
        x = tf.keras.layers.Concatenate(axis=-1, name='concat0')([conv0, x])
        outputs = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same',
                                                  kernel_initializer=initializer,
                                                  use_bias=False, activation='tanh', name='dec_conv0')(x)

        # Return model.
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='Generator')

    def Discriminator(self, source_shape, target_shape):
        """
        Discriminator
        :param source_shape:
        :param target_shape:
        :return: model
        """
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

    def discriminator_loss(self, real_output, fake_output):
        total_loss = tf.reduce_mean(
            -(tf.math.log(real_output + self.epsilon) + tf.math.log(1 - fake_output + self.epsilon)))
        return total_loss

    def generator_loss(self, fake_output, target_images, generated_images):
        gen_loss_GAN = tf.reduce_mean(-tf.math.log(fake_output + self.epsilon))
        gen_loss_L1 = tf.reduce_mean(tf.math.abs(target_images - generated_images))
        total_loss = (gen_loss_GAN * self.gan_weight) + (gen_loss_L1 * self.l1_weight)
        return total_loss

    def generate_images(self, model, test_input, tar):
        # the training=True is intentional here since
        # we want the batch statistics while running the model
        # on the test dataset. If we use training=False, we will get
        # the accumulated statistics learned from the training dataset
        # (which we don't want)
        prediction = model(test_input, training=True)
        plt.figure(figsize=(16, 16))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            gen_output = generator(input_image, training=True)

            dis_real_output = discriminator([input_image, target], training=True)
            dis_generated_output = discriminator([input_image, gen_output], training=True)

            gen_loss = self.generator_loss(dis_generated_output, gen_output, target)
            dis_loss = self.discriminator_loss(dis_real_output, dis_generated_output)

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        dis_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for input_image, target in dataset:
                self.train_step(input_image, target)

            clear_output(wait=True)
            for inp, tar in testset.take(1):
                self.generate_images(generator, inp, tar)

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} took {} sec\n'.format(epoch + 1, time.time() - start))
