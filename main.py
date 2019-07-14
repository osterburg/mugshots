#!/usr/bin/python3

import click

import prep_data as pd
import train_model as tm


@click.command()
@click.option('-e', '--epoch', type=int, default=100)
@click.option('-lr', '--learn_rate', type=float, default=0.0001)
@click.option('-tr', '--train_rate', type=float, default=0.8, help='ratio of training data')
@click.option('-b', '--batch_size', type=int, default=20)
@click.option('-l2', '--l2', type=float, default=0.05, help='L2 regularization')
def main(epoch, learn_rate, train_rate, batch_size, l2):
    # Download dataset and organize it
    pd.prep_data()

    # Pipeline
    trainset = tf.data.Dataset.list_files('data/train/*.png')
    trainset = trainset.shuffle(BUFFER_SIZE)
    trainset = trainset.map(ld.load_train)
    trainset = trainset.batch(1)

    testset = tf.data.Dataset.list_files('data/test/*.png')
    testset = testset.map(ld.oad_test)
    testset = testset.batch(1)

    #
    generator = GAN.Generator((256, 256, 3))
    discriminator = GAN.Discriminator((256, 256, 3), (256, 256, 3))

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.99, epsilon=epsilon)
    dis_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.99, epsilon=epsilon)

    # Checkpoint
    checkpoint_dir = 'data/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer, discriminator_optimizer=dis_optimizer,
                                     generator=generator, discriminator=discriminator)

    # Train
    tm.train(trainset, EPOCHS)

    # Test
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Run the trained model on the entire test dataset
    for inp, tar in testset.take(5):
        generate_images(generator, inp, tar)


if __name__ == '__main__':
    main()
