"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import os
import time
from pathlib import Path

from numpy import mean
from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics.classification import MulticlassJaccardIndex
from argparse import ArgumentParser
import wandb
import datetime as dt

from DataLoader import generate_data_loaders # , calculate_mean
from utils import LABELS, map_id_to_train_id
# from DataVisualizations import disribution_per_chanel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--model_path", type=str, default="model", help="Path to save the model")
    parser.add_argument("--number_of_epochs", type=int, default=10, help="nr of epochs in training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--verbose", type=bool, default=True, help="Print out the training scores or not")
    parser.add_argument("--local_exec", type=bool, default=True, help="Run the training locally or not")
    
    return parser

def _init_wandb(args:ArgumentParser):
    if args.local_exec:
        run = wandb.init(
            # Set the project where this run will be logged
            project="SegmentationTrafficImages",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.number_of_epochs,
                
            },
        )

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    _init_wandb(args)
    # data loading
    wandb.log({"Program Started":dt.datetime.now()})
    train_loader, val_loader = generate_data_loaders(args)
    wandb.log({"Data Loaded":dt.datetime.now()})
    print("Data loaded at ", dt.datetime.now())
    print("Device: ", DEVICE)
    # calculate mean and std of the dataset
    # figures, targets = next(iter(train_loader))
    # figure = disribution_per_chanel(calculate_mean(figures))

    # visualize example images

    # define model
    model = Model().to(DEVICE)

    # define optimizer and loss function (don't forget to ignore class index 255)
    # todo convert to grid optimizer
    lr = args.learning_rate
    num_epochs = args.number_of_epochs
    verbose = args.verbose
    
    # criterion and optimizer for training
    # criterion = MulticlassJaccardIndex(num_classes=34, ignore_index=255, average="macro")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # creterion for validation
    criterion_val_dict = {"CrossEntropy": nn.CrossEntropyLoss(), }#"JaccardIndex": MulticlassJaccardIndex(num_classes=34, ignore_index=255, average="macro")}
    criterion_val_performance = {key: [] for key in criterion_val_dict.keys()}

    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            # ignore labels that are not in test set 
            labels = labels.squeeze(1).long()
            labels = map_id_to_train_id(labels).to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

        epoch_loss = running_loss / len(train_loader)
        
        if verbose:
            wandb.log({"Epoch": (epoch + 1)/num_epochs, "Loss": round(epoch_loss,4)})
            print({"Epoch": (epoch + 1)/num_epochs, "Loss": round(epoch_loss,4)})
            
        
        # validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                # ignore labels that are not in test set 
                labels = labels.squeeze(1).long()
                labels = map_id_to_train_id(labels).to(DEVICE)
                outputs = model(inputs)
                for criterion_name, criterion in criterion_val_dict.items():
                    criterion_val_performance[criterion_name].append(criterion(outputs, labels).detach().item())

            if verbose:
                for criterion_name, criterion_loss in criterion_val_performance.items():
                    wandb.log({f"{criterion_name} Loss": round(mean(criterion_loss),4)})
                    print({f"{criterion_name} Loss": round(mean(criterion_loss),4)})


    # save model
    path_to_model = os.path.join(args.model_path, f"model {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt")
    torch.save(model.state_dict(), path_to_model)

    # visualize some results
    print("Finished at ", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)
