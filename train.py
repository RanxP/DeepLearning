"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
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
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

import wandb
from tqdm import tqdm


from DataLoader import * # , calculate_mean
from utils import LABELS, map_id_to_train_id, train_id_to_name
from DataVisualizations import visualize_criterion
from train_utils import _init_wandb, _print_quda_info, load_model_weights, process_validation_performance, log_dice_loss, ModelEvaluator, save_model

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and validation")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save the model")
    parser.add_argument("--workers", type=int, default=8, help="Path to save the model")
    parser.add_argument("--number_of_epochs", type=int, default=3, help="nr of epochs in training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--verbose", type=bool, default=True, help="Print out the training scores or not")
    parser.add_argument("--cloud_exec", action=BooleanOptionalAction, default=False, help="Run the training locally or not")
    
    parser.add_argument("--figure_size", type=int, default=8, help="height of the figure in pixels described in the power of 2")
    parser.add_argument("--TRANSFORM_STRUCTURE", type= list, default=[TRANSFORM_STRUCTURE], help="Training transformation")
    parser.add_argument("--TRANSFORM_STRUCTURE_VAL", type= list, default=TRANSFORM_STRUCTURE_VAL, help="Validation transformation")
    parser.add_argument("--TRANSFORM_IMAGE", type= list, default=TRANSFORM_IMAGE, help="Image transformation")
    parser.add_argument("--TRANSFORM_MASK", type= list, default=TRANSFORM_MASK, help="Mask transformation")
    
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for the model")
    
    
    return parser

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    _init_wandb(args)
    _print_quda_info
    
    train_loader, val_loader = generate_data_loaders(args)

    #from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    #processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-large")
    #model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")
    model = Model()
    model = load_model_weights(model, "model_best_performance_quijfmub.pth")
    model = model.to(DEVICE)

    # define optimizer and loss function (don't forget to ignore class index 255)
    # todo convert to grid optimizer
    lr = args.learning_rate
    num_epochs = args.number_of_epochs
    verbose = args.verbose
    print("model defined at ", dt.datetime.now())
    
    
    # Define loss criteria to be used
    cross_entropy = nn.CrossEntropyLoss(ignore_index=19,reduction='mean')
    # dice_weighted = MulticlassF1Score(average='weighted',num_classes=20,ignore_index=19)
    # extra loss function to visualize training
    dice = MulticlassF1Score(average=None,num_classes=20,ignore_index=19)
    # criterion and optimizer for training
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = cross_entropy
    wandb.log({"criterion": criterion, "optimizer": "adam"})
    
    # creterion for validation
    criterion_val_dict = {"CrossEntropy": [nn.CrossEntropyLoss(ignore_index=19,reduction='mean'),False], 
                        "Dice": [MulticlassF1Score(average=None,num_classes=20,ignore_index=19),True],
                        "JaccardIndex": [MulticlassJaccardIndex(num_classes=20,ignore_index=19, average="macro"),True]}
    # save model checkpoint 
    ME = ModelEvaluator()

    print("criterion and optimizer defined at ", dt.datetime.now())
    # log model and criterion
    wandb.watch(model,criterion,log="all",log_freq=50)
    # training/validation loop
    for epoch in range(num_epochs):
        # clean cache
        torch.cuda.empty_cache()
        model.to(DEVICE)
        criterion.to(DEVICE)
        dice.to(DEVICE)
        dice_losses = []

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
            # logg dice los of epoch
            dice_losses.append(dice(outputs,labels).detach().cpu())
            
            # Delete variables to free up memory
            del inputs, target, labels, outputs, loss

        epoch_loss = running_loss / len(train_loader)
        if verbose:
            wandb.log({"train": {"Epoch": (epoch + 1)/num_epochs, "CrossEntropy Loss": round(epoch_loss,4)}})
            print({"Epoch": (epoch + 1)/num_epochs, "Loss": round(epoch_loss,4)})
            log_dice_loss(dice_losses,"train")
            
        # clean cache
        torch.cuda.empty_cache()
        model.to(DEVICE)
        # validation loop
        criterion_val_performance = {'loss': {key: [] for key in criterion_val_dict.keys()}, 'outputs': [], 'labels': []}
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                # ignore labels that are not in test set 
                labels = labels.long().squeeze()
                labels = map_id_to_train_id(labels)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                argmax_outputs = torch.argmax(input=outputs,dim=1).to(DEVICE)
                
                #remove_class 255#

                for criterion_name, (criterion_val, one_chanel_prediction)in criterion_val_dict.items():
                    criterion_val = criterion_val.to(DEVICE)
                    if one_chanel_prediction:
                        loss_value = criterion_val(argmax_outputs, labels).detach().cpu()
                    else:
                        loss_value = criterion_val(outputs, labels).detach().item()
                    criterion_val_performance['loss'][criterion_name].append(loss_value)
                criterion_val_performance['outputs'].append(argmax_outputs.cpu())
                criterion_val_performance['labels'].append(labels.cpu())
                
                # Later, when logging or printing:
            
            process_validation_performance(criterion_val_performance)
            # save checkpoint if performance is better
            if (epoch+ 1) % 5 == 0:
                save_model(model, args, f"checkpoint_{epoch}")
            
            if (epoch + 1)/num_epochs > 0.75:
                if ME.best_performace(criterion_val_performance['loss']):
                    save_model(model, args, f"best_performance")
    
    save_model(model, args, "final")
        
    # visualize some results
    print("Finished at ", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)
