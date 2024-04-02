from utils import LABELS, map_id_to_train_id, train_id_to_name
from DataVisualizations import visualize_criterion


import os
import time
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import mean
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

import wandb
from tqdm import tqdm
def _init_wandb(args):
    run = wandb.init(
        # Set the project where this run will be logged
        project="SegmentationTrafficImages",
        # Track hyperparameters and run metadata
        config=args.__dict__)
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
 
 
def process_validation_performance(criterion_val_performance:dict):

    criterion_losses = criterion_val_performance['loss']
    output_tensors = criterion_val_performance['outputs']
    label_tensors = criterion_val_performance['labels']
    
    # Calculate the mean loss for each criterion
    # add jaccard index
    log_dice_loss(criterion_losses["Dice"],"val")
    
    # Find the index of the maximum loss
    loss_entropy= criterion_losses["CrossEntropy"]
    wandb.log({"val":{"CrossEntropy Loss": round(mean(loss_entropy),4)}})
    
    max_loss_index = loss_entropy.index(max(loss_entropy))

    # Select the output and label tensor of the highest loss
    max_loss_output = output_tensors[max_loss_index]
    max_loss_label = label_tensors[max_loss_index]

    
    visualize_criterion(baseline=max_loss_label, prediction=max_loss_output, 
                        loss=max(loss_entropy), criterion_name="CrossEntropy")
    
def log_dice_loss(list_of_losses, wandb_group:str="train"):
    dice_stack = torch.stack(list_of_losses)
    dice_loss_per_class= torch.mean(dice_stack,dim=0)
    wandb.log({f"{wandb_group}":{"mean Dice Loss": round(torch.mean(dice_loss_per_class).item(),4)}})
    print({"mean Dice Loss": round(torch.mean(dice_loss_per_class).item(),4)})
    
    for train_id, dice in enumerate(dice_loss_per_class):
        wandb.log({f"{wandb_group}":{f"Dice_{train_id_to_name(train_id)}": round(dice.item(),4)}})
        
        
class ModelEvaluator:
    
    def best_performace(self,criterion_losses) -> bool:
        
        loss_entropy= criterion_losses["CrossEntropy"]
        if not hasattr(self, "best_score"):
            self.best_score = mean(loss_entropy)
            return False
            
        if self.best_score > mean(loss_entropy):
            self.best_score = mean(loss_entropy)
            return True        
    
def save_model(model, args, name_parameter:str):
    model_dir = os.path.join(os.getcwd(), args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Create a timestamp for the saved model
    model_filename = f"/model_{name_parameter}_{wandb.run._run_id}.pth"

    # Save the model
    torch.save(model.state_dict(), model_dir + model_filename)
    
def _hot_load_model(model ,model_path:str):
    full_model_path = os.path.join(os.getcwd(), model_path)
    model.load_state_dict(torch.load(full_model_path))
    model.eval()
    return model

        