"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics.classification import MulticlassJaccardIndex
from argparse import ArgumentParser


from DataLoader import generate_data_loaders, calculate_mean
from DataVisualizations import disribution_per_chanel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    
    train_loader, val_loader = generate_data_loaders(args)
    
    # calculate mean and std of the dataset
    # figures, targets = next(iter(train_loader))
    # figure = disribution_per_chanel(calculate_mean(figures))

    # visualize example images

    # define model
    model = Model().to(DEVICE)

    # define optimizer and loss function (don't forget to ignore class index 255)
    # todo convert to grid optimizer
    lr = 0.001
    num_epochs = 10 
    verbose = True
    criterion = MulticlassJaccardIndex(num_classes=34, ignore_index=255, average="macro")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            labels.unsqueeze(1).long().shape
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

        epoch_loss = running_loss / len(train_loader)
        
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        


    # save model


    # visualize some results

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)
