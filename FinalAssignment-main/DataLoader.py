from unittest.mock import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import numpy as np

CHANNEL_MEANS = [0.22460080459713935, 0.26541953831911086, 0.22537076537098202]
CHANNEL_STDS = [0.019116874995935115, 0.02040196749932445, 0.02062898499852692]


TRANSFORM_TEST =  transforms.Compose([
    transforms.Resize((256, 256)), # resize
    # data transformation
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    ])
TRANSFORM_TRAIN =  transforms.Compose([
    transforms.Resize((256, 256)), # resize
    # data transformation
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=CHANNEL_MEANS, std = CHANNEL_STDS)
    ])


def generate_data_loaders(args) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    trainset = Cityscapes(root = args.data_path, split='train', mode='fine', 
                          transform=TRANSFORM_TRAIN, target_transform = TRANSFORM_TEST,
                          target_type='semantic')
    train_subset, val_subset = torch.utils.data.random_split(trainset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1))
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=10,
                                            shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=10,
                                            shuffle=True, num_workers=2)
    
    
    return trainloader, valloader



    