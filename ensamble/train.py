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

from DataLoader import * # , calculate_mean
from utils import LABELS, map_id_to_train_id, train_id_to_name, remove_classes_from_tensor
from DataVisualizations import visualize_criterion
from support_train import _init_wandb, _print_quda_info, process_validation_performance, log_dice_loss, ModelEvaluator, save_model

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
        decoders.append(decoder)
        optimizers.append(optim.Adam(decoder.parameters(), lr=wandb.config.learning_rate))
    return classes_to_ignore, decoders, optimizers

    
def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save the model")
    parser.add_argument("--workers", type=int, default=8, help="Path to save the model")
    parser.add_argument("--number_of_epochs", type=int, default=3, help="nr of epochs in training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--verbose", type=bool, default=True, help="Print out the training scores or not")
    parser.add_argument("--cloud_exec", action=BooleanOptionalAction, default=False, help="Run the training locally or not")
    
    parser.add_argument("--figure_size", type=tuple[int,int], default=IMG_SIZE, help="Width of the figure in pixels")
    parser.add_argument("--TRANSFORM_STRUCTURE", type= list, default=[TRANSFORM_STRUCTURE], help="Training transformation")
    parser.add_argument("--TRANSFORM_STRUCTURE_VAL", type= list, default=TRANSFORM_STRUCTURE_VAL, help="Validation transformation")
    parser.add_argument("--TRANSFORM_IMAGE", type= list, default=TRANSFORM_IMAGE, help="Image transformation")
    parser.add_argument("--TRANSFORM_MASK", type= list, default=TRANSFORM_MASK, help="Mask transformation")
    
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for the model")
    
    
    return parser

def load_encoder_weights(encoder,model_path):
    full_model_path = os.path.join(os.getcwd(), model_path)
    pretrained_dict = (torch.load(full_model_path))
    model_dict = encoder.state_dict()
        # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    encoder.load_state_dict(pretrained_dict)


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    _init_wandb(args)
    _print_quda_info
    
    train_loader, val_loader = generate_data_loaders(args)

    # define model
    encoder = pre_trained_encoder()
    load_encoder_weights(encoder, "model\model_best_performance_quijfmub.pth")

    classes_to_ignore, decoders, optimizers = create_decoders(3)
    model =  EnsambleModel(encoder, decoders)
    model.freeze_encoder()
    # torch.compile(model)
    model = model.to(DEVICE)
    
    # Define loss criteria to be used
    criterion = nn.CrossEntropyLoss(ignore_index=19,reduction='mean')
    dice = MulticlassF1Score(average=None,num_classes=20,ignore_index=19).to(DEVICE)
    # crcriterioneterion for validation
    criterion_val_dict = {"CrossEntropy": [nn.CrossEntropyLoss(ignore_index=19,reduction='mean'),False], 
                        "Dice": [MulticlassF1Score(average=None,num_classes=20,ignore_index=19),True],}

    # log model and criterion
    wandb.watch(model,criterion,log="all",log_freq=50)
    # training/validation loop
    for epoch in range(wandb.config.number_of_epochs):
        # clean cache
        torch.cuda.empty_cache()
        model.to(DEVICE)
        criterion.to(DEVICE)

        running_loss = 0.0
        dice_decoder_losses = [[],[],[]]
        model.train()
        # training loop
        for inputs, target in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{wandb.config.number_of_epochs}"):
            inputs = inputs.to(DEVICE)
            # ignore labels that are not in test set 
            target = target.long().squeeze()
            target = map_id_to_train_id(target)
            outputs = model(inputs)
            # print(outputs.shape)
            dice_losses = []
            # multiple outputs 
            for i, output in enumerate(outputs):
                total_loss = 0
                print(output.shape)
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
                
                dice_decoder_losses[i].append(dice(output,decoder_specific_lables).detach().cpu())
                
                
                
                total_loss += loss.item()
            
            
            running_loss += total_loss / 3
            print(running_loss)
            
            outputs_tensor = torch.stack(outputs)
            mean_outputs = torch.mean(outputs_tensor, dim=0, keepdim=False).to(DEVICE)
            print(mean_outputs.shape)
            target = target.to(DEVICE)
            dice_losses.append(dice(mean_outputs,target).detach().cpu())

            ensamble_output = torch.argmax(input=mean_outputs,dim=0).to(DEVICE)
            print(ensamble_output.shape)
            
            # logg dice los of epoch
            
            # Delete variables to free up memory
            del inputs, target, decoder_specific_lables, outputs, loss
            
        if wandb.config.verbose:
            wandb.log({"train": {"Epoch": (epoch + 1)/wandb.config.number_of_epochs, "CrossEntropy Loss": round(running_loss/35,4)}})
            # print({"Epoch": (epoch + 1)/num_epochs, "Loss": round(epoch_loss,4)})
            log_dice_loss(dice_losses,"train")
            for i, dice_loss in enumerate(dice_decoder_losses):
                log_dice_loss(dice_loss,f"train_decoder_{classes_to_ignore[i]}")
            
        # clean cache
        # torch.cuda.empty_cache()
        # model.to(DEVICE)
        # # validation loop
        # criterion_val_performance = {'loss': {key: [] for key in criterion_val_dict.keys()}, 'outputs': [], 'labels': []}
        # model.eval()
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs = inputs.to(DEVICE)
        #         # ignore labels that are not in test set 
        #         labels = labels.long().squeeze()
        #         labels = map_id_to_train_id(labels)
        #         labels = labels.to(DEVICE)
                
        #         outputs = model(inputs)
        #         argmax_outputs = torch.argmax(input=outputs,dim=1).to(DEVICE)
                
        #         #remove_class 255#

        #         for criterion_name, (criterion_val, one_chanel_prediction)in criterion_val_dict.items():
        #             criterion_val = criterion_val.to(DEVICE)
        #             if one_chanel_prediction:
        #                 loss_value = criterion_val(argmax_outputs, labels).detach().cpu()
        #             else:
        #                 loss_value = criterion_val(outputs, labels).detach().item()
        #             criterion_val_performance['loss'][criterion_name].append(loss_value)
        #         criterion_val_performance['outputs'].append(argmax_outputs.cpu())
        #         criterion_val_performance['labels'].append(labels.cpu())
                
        #         # Later, when logging or printing:
            
        #     process_validation_performance(criterion_val_performance)
        #     # save checkpoint if performance is better
        #     if (epoch + 1)/num_epochs > 0.75:
        #         if ME.best_performace(criterion_val_performance['loss']):
        #             save_model(model, args, f"best_performance")
    
    save_model(model, args, "final")
        
    # visualize some results
    print("Finished at ", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_known_args()[0]
    main(args)