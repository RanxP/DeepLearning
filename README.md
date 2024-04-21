# Final Assignment

This repository serves as the my assignment for the 5LSM0 final assignment.
This assignment is a part of the 5LSM0 course. It involves working with the Cityscapes dataset and training a neural network. The assignment contributes to 50% of your final grade.
it build on the starter kit of: 
### Authors

- T.J.M. Jaspers
- C.H.B. Claessens
- C.H.J. Kusters

## Getting Started

Welcome to the repository let me walk you trough what is what from from top to bottom. 

Folders:
**Data** - contains my local development data
**ensamble** - all ensamble specific files
**ensamble_handin/** - an alterd u-net that consumes less memory is at the core of this file. as u net that keeps all memory cached is too large for hand in 
    **Ensamble_handin/visualize_distributions.py** constructs the visuals seen for figure 1/3 and the ROC scores.
    re rest of the files are used for submission, but are also needed for functioning of this file. 
**model** - contains my local development models


The main folder is everything i needed to train one u-net + shared functions with training an ensamble.

- DataLoader Contains dataloaders, these are constructed with the help of val_loader and Train loader which have their own transformations

- Data Visualization - Contains functions that are used to plot classifications, such as temp_plot

- **model.py:** Defines the neural network architecture. is ment for training

- **train.py** contains the training logging and evaluation loop of the single u net.

- **train_utils.py** contains the functions needed in training and shared across single and ensamble u net

- **utils** Provided utils file with extra functionality to convert classes

Function files
run_container. specifies what file should be trained. has two possible directories
    1. train.py 
    2. ensamble/train.py

run_results, quick way to run results without wandb, expects either 
    1. ensamble_handin/visualize_distriutions.py 
    2. visualize_distriutions_one.py


### alterations by 
Ranx Peeters
