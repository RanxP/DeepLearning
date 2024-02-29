"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
from argparse import ArgumentParser


from DataLoader import generate_data_loaders, calculate_mean
from DataVisualizations import disribution_per_chanel


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    
    trainloader, valloader, testloader = generate_data_loaders(args)
    
    # calculate mean and std of the dataset
    figure : plt = disribution_per_chanel(calculate_mean(trainloader))

    # visualize example images

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)


    # training/validation loop


    # save model


    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)
