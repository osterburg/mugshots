#!/usr/bin/python3

import click
import prep_data
import load_data
import train_model


@click.command()
@click.option('-e', '--epoch', type=int, default=100)
@click.option('-lr', '--learn_rate', type=float, default=0.0001)
@click.option('-tr', '--train_rate', type=float, default=0.8, help='ratio of training data')
@click.option('-b', '--batch_size', type=int, default=20)
@click.option('-l2', '--l2', type=float, default=0.05, help='L2 regularization')
def main(epoch, learn_rate, train_rate, batch_size, l2):
    prep_data.prep_data()
    # unet = model.UNet(classes=2)
    # unet.train(parser)


if __name__ == '__main__':
  main()
