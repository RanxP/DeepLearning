
from unittest.mock import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import numpy as np

def generate_transform(means = None, stds= None):
    if isinstance(means, list) and isinstance(stds, list):
        transform =  transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=means, std = stds)
            ])
    else:
        transform =  transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32)
            ])
    return transform



def calculate_mean(dataset):
    mean_per_image_r, mean_per_image_g, mean_per_image_b = [], [], []

    for image, _ in dataset:
        mean_per_image_r.append(torch.mean(image[0,:,:]).tolist())
        mean_per_image_g.append(torch.mean(image[1,:,:]).tolist())
        mean_per_image_b.append(torch.mean(image[2,:,:]).tolist())

    return mean_per_image_r, mean_per_image_g, mean_per_image_b


def generate_data_loaders(args) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    trainset = Cityscapes(root = args.data_path, split='train', mode='fine', 
                          transform=generate_transform(), 
                          target_type='semantic')
    train_subset, val_subset = torch.utils.data.random_split(trainset, [40000, 10000],
                                            generator=torch.Generator().manual_seed(1))

    mean_r, mean_g, mean_b = calculate_mean(train_subset)

    means = [np.mean(mean_r),np.mean(mean_g),np.mean(mean_b)]
    stds = [np.std(mean_r),np.std(mean_g),np.std(mean_b)]


    trainset = Cityscapes(root = args.data_path, split='train', mode='fine', 
                          transform=generate_transform(), 
                          target_type='semantic')
    
    train_subset, val_subset = torch.utils.data.random_split(trainset, [40000, 10000],
                                            generator=torch.Generator().manual_seed(1))

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=10,
                                            shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=10,
                                            shuffle=True, num_workers=2)

    testset = Cityscapes(root = args.data_path, split='test', mode='fine', 
                          transform=generate_transform(), 
                          target_type='semantic')
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                            shuffle=False, num_workers=2)
    
    
    return trainloader, valloader, testloader

if __name__ == "__main__":
    DATA_DIR = "data"
    generate_data_loaders()


    