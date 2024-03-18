
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import wandb

    # Imports PIL module
import numpy as np
import os    
import PIL
from utils import LABELS   

def transform_tensor_to_class_collor(target) -> np.array:
    """Visualize the cityscapes label
    input: Tensor of shape (batch_size, height, width) with values from 0 to 18
    output: None
    """

    # creating a image object (new image object) with
    im = PIL.Image.new(mode="RGB", size=(target.shape[1], target.shape[2]))
    im = np.array(im)
    for label in LABELS:
        # replace the value in the tensor with the train id if the value in the input tensor is equal to the id of the label
        im[target.squeeze() == label.id] = label.color
    
    return im

def transform_list_to_class_color(target) -> np.array:
    """Visualize the cityscapes label
    input: Tensor of shape (batch_size, height, width) with values from 0 to 18
    output: None
    """

    # creating a image object (new image object) with
    im = PIL.Image.new(mode="RGB", size=(target.shape[0], target.shape[1]))
    im = np.array(im)
    for label in LABELS:
        # replace the value in the tensor with the train id if the value in the input tensor is equal to the id of the label
        im[target == label.id] = label.color
    
    return im

    
def visualize_criterion(baseline, prediction, loss, criterion_name:str):
    """Visualize the difference between the baseline and the prediction
    input: baseline: Tensor of shape (height, width) with values from 0 to 18
    input: prediction: Tensor of shape (height, width) with values from 0 to 18
    input: loss: float
    input: criterion_name: str
    output: None
    """
    # create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 15))
    # visualize the baseline
    axs[0].imshow(transform_list_to_class_color(baseline))
    axs[0].set_title('Baseline')
    axs[0].axis("off")
    # visualize the prediction
    axs[1].imshow(transform_list_to_class_color(prediction))
    axs[1].set_title('Prediction')
    axs[1].axis("off")
    # visualize the loss
    plt.suptitle(f'{criterion_name} Loss: {loss}')

    # Save the figure to a file
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "temp_plot.png")
    fig.savefig(file_path)
    plt.close(fig)

    # Log the image file to wandb
    wandb.log({f"{criterion_name} visualization": [wandb.Image("temp_plot.png", caption=f"{criterion_name} Loss: {loss}")]})
