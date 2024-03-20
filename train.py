"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from ast import arg
import os
import time
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction

from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex

import wandb
from numpy import argmax, mean
from tqdm import tqdm


from DataLoader import generate_data_loaders # , calculate_mean
from utils import LABELS, map_id_to_train_id
from DataVisualizations import visualize_criterion

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training and validation")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save the model")
    parser.add_argument("--workers", type=int, default=8, help="Path to save the model")
    parser.add_argument("--number_of_epochs", type=int, default=1, help="nr of epochs in training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--verbose", type=bool, default=True, help="Print out the training scores or not")
    parser.add_argument("--cloud_exec", action=BooleanOptionalAction, default=False, help="Run the training locally or not")
    
    return parser

def _init_wandb():
    run = wandb.init(
        # Set the project where this run will be logged
        project="SegmentationTrafficImages",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.number_of_epochs,
        })
    time.sleep(2)
    wandb.log({"Program Started":dt.datetime.now()})

def _print_quda_info():
    if torch.cuda.is_available():
        # torch.cuda.set_device(0)
        print("Current CUDA device: ", torch.cuda.current_device())
        print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Device variable: ", DEVICE)
    else:
        print("CUDA is not available. Using CPU.")
        
def _hot_load_model(model :Model ,model_path:str):
    full_model_path = os.path.join(os.getcwd(), model_path)
    model.load_state_dict(torch.load(full_model_path))
    model.eval()
    return model

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    _init_wandb()
    _print_quda_info
    
    train_loader, val_loader = generate_data_loaders(args)
    
    # visualize example images

    # define model
    model = Model().to(DEVICE)
    print(args.cloud_exec)
    # if not args.cloud_exec:
        # model = _hot_load_model(model, Path("model/model.pt"))

    # define optimizer and loss function (don't forget to ignore class index 255)
    # todo convert to grid optimizer
    lr = args.learning_rate
    num_epochs = args.number_of_epochs
    verbose = args.verbose
    print("model defined at ", dt.datetime.now())
    
    # criterion and optimizer for training
    # criterion = MulticlassJaccardIndex(num_classes=34, ignore_index=255, average="macro")
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # creterion for validation
    criterion_val_dict = {"CrossEntropy": nn.CrossEntropyLoss(ignore_index=255,reduction='mean') }#"JaccardIndex": MulticlassJaccardIndex(num_classes=34, ignore_index=255, average="macro")}
    
    print("criterion and optimizer defined at ", dt.datetime.now())
    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        # training loop
        for inputs, target in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(DEVICE)
            # ignore labels that are not in test set 
            target = target.long().squeeze()
            target = map_id_to_train_id(target)
            labels = target.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            # Delete variables to free up memory
            del inputs, target, labels, outputs, loss

        epoch_loss = running_loss / len(train_loader)
        
        if verbose:
            wandb.log({"Epoch": (epoch + 1)/num_epochs, "Loss": round(epoch_loss,4)})
            print({"Epoch": (epoch + 1)/num_epochs, "Loss": round(epoch_loss,4)})
            
            
        # validation loop
        criterion_val_performance = {key: {'loss': [], 'outputs': [], 'labels': []} for key in criterion_val_dict.keys()}
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                # ignore labels that are not in test set 
                labels = labels.long().squeeze()
                labels = map_id_to_train_id(labels)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                argmax_outputs = torch.argmax(input=outputs,dim=1)
                
                
                for criterion_name, criterion in criterion_val_dict.items():
                    loss_value = criterion(outputs, labels).detach().item()
                    criterion_val_performance[criterion_name]['loss'].append(loss_value)
                    criterion_val_performance[criterion_name]['outputs'].append(argmax_outputs.cpu())
                    criterion_val_performance[criterion_name]['labels'].append(labels.cpu())
                # print(argmax_outputs.shape, labels.shape)
                # print(loss_value)
                # print(criterion_val_performance[criterion_name]['loss'])
                
                # Later, when logging or printing:
            if verbose:
                try:
                    process_validation_performance(criterion_val_performance)
                except Exception as e:
                    print("Error in process_validation_performance: ", e)
                        

    # save model
    model_dir = os.path.join(os.getcwd(), args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Create a timestamp for the saved model
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H:%M')
    model_filename = f"model_{timestamp}.pth"

    # Create the full path for the saved model
    model_path = os.path.join(model_dir, model_filename)

    # Save the model
    torch.save(model.state_dict(), model_dir + model_filename)

    # visualize some results
    print("Finished at ", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def process_validation_performance(criterion_val_performance:dict):
    for criterion_name, performance in criterion_val_performance.items():
        criterion_loss = performance['loss']
        output_tensors = performance['outputs']
        label_tensors = performance['labels']
        
        wandb.log({f"{criterion_name} Loss": round(mean(criterion_loss),4)})
        print({f"{criterion_name} Loss": round(mean(criterion_loss),4)})
        
            # Find the index of the maximum loss
        max_loss_index = criterion_loss.index(max(criterion_loss))

        # Select the output and label tensor of the highest loss
        max_loss_output = output_tensors[max_loss_index]
        max_loss_label = label_tensors[max_loss_index]

        
        visualize_criterion(baseline=max_loss_label, prediction=max_loss_output, 
                            loss=max(criterion_loss), criterion_name=criterion_name)

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)
