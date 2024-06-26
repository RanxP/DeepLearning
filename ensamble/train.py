"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from json import encoder
import os
import time
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction

from matplotlib import pyplot as plt
import seaborn as sns
from numpy import var


from ensamble_model import EnsambleModel, standalone_decoder, pre_trained_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

import wandb
from tqdm import tqdm

import sys
sys.path.insert(0, os.getcwd())

from DataLoader import *
from utils import LABELS, map_id_to_train_id, train_id_to_name, remove_classes_from_tensor
from DataVisualizations import visualize_criterion
from train_utils import _init_wandb, _print_quda_info, load_model_weights, log_dice_loss, ModelEvaluator, save_model

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_decoders(nr_decoders:int):
    decoders = []
    classes_to_ignore = []
    optimizers = []
    if 18 % nr_decoders != 0:
        raise Warning("Number of decoders must be a factor of 18")
    for i in range(nr_decoders):
        begin_class = i * int(18 / nr_decoders)
        end_class = (i + 1) * int(18 / nr_decoders)
        classes_to_ignore.append( list(range(begin_class,end_class)))
        
        decoder = standalone_decoder().to(DEVICE)
        decoder = decoder.init_weights()
        decoders.append(decoder)
        optimizers.append(optim.Adam(decoder.parameters(), lr=wandb.config.learning_rate))
    return classes_to_ignore, decoders, optimizers

    
def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save the model")
    parser.add_argument("--workers", type=int, default=6, help="Path to save the model")
    parser.add_argument("--number_of_epochs", type=int, default=3, help="nr of epochs in training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--verbose", type=bool, default=True, help="Print out the training scores or not")
    parser.add_argument("--cloud_exec", action=BooleanOptionalAction, default=False, help="Run the training locally or not")
    
    parser.add_argument("--figure_size", type=int, default=7, help="height of the figure in pixels described in the power of 2")
    parser.add_argument("--TRANSFORM_STRUCTURE", type= list, default=TRANSFORM_STRUCTURE, help="Training transformation")
    parser.add_argument("--TRANSFORM_STRUCTURE_VAL", type= list, default=TRANSFORM_STRUCTURE_VAL, help="Validation transformation")
    parser.add_argument("--TRANSFORM_IMAGE", type= list, default=TRANSFORM_IMAGE, help="Image transformation")
    parser.add_argument("--TRANSFORM_MASK", type= list, default=TRANSFORM_MASK, help="Mask transformation")
    
    parser.add_argument("--nr_of_decoders", type=int, default=6, help="Dropout rate for the model")
    parser.add_argument("--wandb_mode", type=str, default="offline", help="Wandb mode")
    
    
    return parser

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    _init_wandb(args)
    _print_quda_info(DEVICE=DEVICE)
    
    train_loader, val_loader = generate_data_loaders(args)

    # define model
    encoder = pre_trained_encoder()
    encoder = load_model_weights(encoder, "model_checkpoint_24_uaij0fix.pth")
    
    classes_to_ignore, decoders, optimizers = create_decoders(wandb.config.nr_of_decoders)
    model =  EnsambleModel(encoder, decoders)
    #load_encoder_weights(model, "model_final_vhb12qyp.pth")
    model.freeze_encoder()
    # torch.compile(model)
    model = model.to(DEVICE)
    
    # Define loss criteria to be used
    criterion = nn.CrossEntropyLoss(ignore_index=19,reduction='mean').to(DEVICE)
    dice = MulticlassF1Score(average=None,num_classes=20,ignore_index=19).to(DEVICE)
    # crcriterioneterion for validation
    criterion_val_dict = {"CrossEntropy": [nn.CrossEntropyLoss(ignore_index=19,reduction='mean'),False], 
                        "Dice": [MulticlassF1Score(average=None,num_classes=20,ignore_index=19),True],}

    train_total_known_classes_activation = []
    train_total_unknown_classes_activation = []
    train_total_mean_softmax_score_of_image = []
    val_total_known_classes_activation = []
    val_total_unknown_classes_activation = []
    val_total_mean_softmax_score_of_image = []
    # log model and criterion
    wandb.watch(model,criterion,log="all",log_freq=50)
    min_dice_loss = 0
    # training/validation loop
    for epoch in range(wandb.config.number_of_epochs):
        # clean cache
        torch.cuda.empty_cache()

        running_loss = 0.0
        dice_decoder_losses = [[]] * wandb.config.nr_of_decoders
        dice_losses_train = []
        model.eval()
        # training loop
        for inputs, target in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{wandb.config.number_of_epochs}"):
            torch.cuda.empty_cache()
            inputs = inputs.to(DEVICE)
            # ignore labels that are not in test set 
            target = target.long().squeeze()
            target = map_id_to_train_id(target)
            outputs = model(inputs)
            del inputs
            
            total_loss = 0
            # multiple outputs 
            for i, output in enumerate(outputs):
                torch.cuda.empty_cache()
                output = output.to(DEVICE)
                # convert abels to exclude classes
                decoder_specific_lables = remove_classes_from_tensor(target, classes_to_ignore[i])
                decoder_specific_lables = decoder_specific_lables.to(DEVICE)
                # devise los for one specific decoder
                loss = criterion(output,decoder_specific_lables)
                # take a step for only one decoder ? 
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
            
                total_loss += loss.item()
                del output, decoder_specific_lables, loss
                

            
            running_loss += total_loss / 3
            print(running_loss)
            with torch.no_grad():
                outputs_tensor = torch.stack(outputs) # shape (3,4,20,512,1024)
                normalized_outputs = F.softmax(outputs_tensor, dim=2) # checked is correct
                mean_outputs = torch.mean(normalized_outputs, dim=0, keepdim=False).to(DEVICE)
                # var_outputs = torch.var(normalized_outputs, dim=0, keepdim=False)
                # ensamble_output = torch.argmax(input=mean_outputs,dim=1)
                
                target = target.to(DEVICE)
                dice_losses_train.append(dice(mean_outputs,target).detach().cpu())
                results = calibrate_activation(mean_outputs, target)
                train_total_known_classes_activation.append(results['known_classes_activation'])
                train_total_unknown_classes_activation.append(results['unknown_classes_activation'])
                train_total_mean_softmax_score_of_image.append(results['mean_softmax_score_of_image'])
                # calculate activation of known and unknown classes
                # Delete variables to free up memory
                del target, mean_outputs, outputs, outputs_tensor, normalized_outputs, results
        
        if wandb.config.verbose:
            wandb.log({"train": {"Epoch": (epoch + 1)/wandb.config.number_of_epochs, "CrossEntropy Loss": round(running_loss/35,4)}})

            mean_dice_loss = log_dice_loss(dice_losses_train,"train")
            if mean_dice_loss > min_dice_loss:
                min_dice_loss = mean_dice_loss
                save_model(model, args, f"best_train_performance")
                # save_model(model, args, f"best_performance_epoch_{epoch+1}")

            # for i, dice_loss in enumerate(dice_decoder_losses):
            #     log_dice_loss(dice_loss,f"train_decoder_{classes_to_ignore[i]}")
                
        # clean cache
        dice_losses_val = []
        running_val_loss = 0
        with torch.no_grad():
            for inputs, target in tqdm(val_loader, desc=f"Training epoch {epoch+1}/{wandb.config.number_of_epochs}"):
                total_loss = 0
                torch.cuda.empty_cache()
                inputs = inputs.to(DEVICE)
                # ignore labels that are not in test set 
                target = target.long().squeeze()
                target = map_id_to_train_id(target)
                outputs = model(inputs)
                # multiple outputs 
                for i, output in enumerate(outputs):
                    output = output.to(DEVICE)
                    # convert abels to exclude classes
                    decoder_specific_lables = remove_classes_from_tensor(target, classes_to_ignore[i])
                    decoder_specific_lables = decoder_specific_lables.to(DEVICE)
                    # devise los for one specific decoder
                    loss = criterion(output,decoder_specific_lables)
                    dice_decoder_losses[i].append(dice(output,decoder_specific_lables).detach().cpu())
                    total_loss += loss.item()
                    del output, decoder_specific_lables, loss
                running_val_loss += total_loss / 3
                outputs_tensor = torch.stack(outputs) # shape (3,4,20,512,1024)
                normalized_outputs = F.softmax(outputs_tensor, dim=2) # checked is correct
                mean_outputs = torch.mean(normalized_outputs, dim=0, keepdim=False).to(DEVICE)
                # var_outputs = torch.var(normalized_outputs, dim=0, keepdim=False)
                # ensamble_output = torch.argmax(input=mean_outputs,dim=1)
                target = target.to(DEVICE)
                dice_losses_val.append(dice(mean_outputs,target).detach().cpu())
                results = calibrate_activation(mean_outputs, target)
                val_total_known_classes_activation.append(results['known_classes_activation'])
                val_total_unknown_classes_activation.append(results['unknown_classes_activation'])
                val_total_mean_softmax_score_of_image.append(results['mean_softmax_score_of_image'])
                # calculate activation of known and unknown classes
                # Delete variables to free up memory
                del target, mean_outputs, outputs, outputs_tensor, normalized_outputs, results
            
        if wandb.config.verbose:
            wandb.log({"val": {"CrossEntropy Loss": round(running_loss/35,4)}})

            # log mean_softmax_score_of_image and activation of known and unknown classes
            wandb.log({"train": {"mean_softmax_score_of_image": round(torch.mean(torch.tensor(train_total_mean_softmax_score_of_image)).item(),4),
                                "known_classes_activation": round(torch.mean(torch.tensor(train_total_known_classes_activation)).item(),4),
                                "unknown_classes_activation": round(torch.mean(torch.tensor(train_total_unknown_classes_activation)).item(),4)}})
            wandb.log({"val": {"mean_softmax_score_of_image": round(torch.mean(torch.tensor(val_total_mean_softmax_score_of_image)).item(),4),
                                "known_classes_activation": round(torch.mean(torch.tensor(val_total_known_classes_activation)).item(),4),
                                "unknown_classes_activation": round(torch.mean(torch.tensor(val_total_unknown_classes_activation)).item(),4)}})
            # visualize the distribution of the activations
            fig, ax = plt.subplots(2,1, figsize=(5,10))
            
            # First subplot for training data
            ax[0].set_title("Training")
            sns.histplot(train_total_known_classes_activation, bins=100, ax=ax[0], label='Known classes activation', kde=True)
            sns.histplot(train_total_unknown_classes_activation, bins=100, ax=ax[0], label='Unknown classes activation', kde=True)
            ax[0].legend()

            # Second subplot for validation data
            ax[1].set_title("Validation")
            sns.histplot(val_total_known_classes_activation, bins=100, ax=ax[1], label='Known classes activation', kde=True)
            sns.histplot(val_total_unknown_classes_activation, bins=100, ax=ax[1], label='Unknown classes activation', kde=True)
            ax[1].legend()

            # Save the figure
            wandb.log({"activation on known versus unknown classes seaborn": wandb.Image(fig)})
            plt.close(fig)
            
            
            mean_dice_loss = log_dice_loss(dice_losses_val,"val")
            if mean_dice_loss > min_dice_loss:
                min_dice_loss = mean_dice_loss
                save_model(model, args, f"best_performance")
                # save_model(model, args, f"best_performance_epoch_{epoch+1}")

            for i, dice_loss in enumerate(dice_decoder_losses):
                log_dice_loss(dice_loss,f"val_decoder_{classes_to_ignore[i]}")
            
        # visualize distributions of activations
    
    save_model(model, args, "final")
        
    # visualize some results
    print("Finished at ", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    
def calibrate_activation(mean_outputs, target):
    target = target.cpu()
    mean_outputs = mean_outputs.cpu()
    known_indices = (target != 19)
    unknown_indices = (target == 19)

    # Compute the mean activation of the known and unknown classes
    activation_score_per_image, prediction_per_image = torch.max(mean_outputs.permute(0,2,3,1)[known_indices],dim=1) # dim is checked 
    activation_score_per_image_unknown, prediction_per_image_unknown = torch.max(mean_outputs.permute(0,2,3,1)[unknown_indices],dim=1)
    softmax_score_per_pixel, _ = torch.max(mean_outputs.permute(0,2,3,1), dim=3)
    print(activation_score_per_image.shape)
    print(softmax_score_per_pixel.shape, softmax_score_per_pixel)

    known_classes_activation = torch.mean(activation_score_per_image).item()
    unknown_classes_activation = torch.mean(activation_score_per_image_unknown).item()
    mean_softmax_score_of_image = torch.mean(softmax_score_per_pixel).item()
    
    return {
        'mean_softmax_score_of_image': mean_softmax_score_of_image,
        'known_classes_activation': known_classes_activation,
        'unknown_classes_activation': unknown_classes_activation
    }

    

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)
