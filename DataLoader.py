from unittest.mock import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import numpy as np
from pathlib import Path

CHANNEL_MEANS = [0.22460080459713935, 0.26541953831911086, 0.22537076537098202]
CHANNEL_STDS = [0.019116874995935115, 0.02040196749932445, 0.02062898499852692]

IMG_SIZE = (512, 512)
TRANSFORM_MASK =  transforms.Compose([
    transforms.Resize(size=IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS), # resize
    # data transformation
    transforms.PILToTensor(),
    #.ConvertImageDtype(torch.float32),
    ])
TRANSFORM_IMAGE =  transforms.Compose([
    transforms.Resize(size=IMG_SIZE,interpolation=transforms.InterpolationMode.LANCZOS), # resize
    # data transformation
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=CHANNEL_MEANS, std = CHANNEL_STDS)
    ])


def generate_data_loaders(args) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    trainset = Cityscapes(root = Path(args.data_path), split='train', mode='fine', 
                          transform=TRANSFORM_IMAGE, target_transform = TRANSFORM_MASK,
                          target_type='semantic')
    train_subset, val_subset = torch.utils.data.random_split(trainset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1))
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)
    
    
    return trainloader, valloader



    