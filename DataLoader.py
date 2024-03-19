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
import wandb

CHANNEL_MEANS = [0.485, 0.456, 0.406]
CHANNEL_STDS = [0.229, 0.224, 0.225]

IMG_SIZE = (512,1024)

# ideas 5 crop
# perspective transform
#rotation 10 % each direction

# color or gray scale transformations

TRANSFORM_STRUCTURE = [
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS)
    # transforms.RandomCrop(size=IMG_SIZE)
    # transforms.RandomRotation(degrees=10),
    ]

TRANSFORM_IMAGE =  transforms.Compose(
    TRANSFORM_STRUCTURE + [
    # data transformation
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    # transforms.Normalize(mean=CHANNEL_MEANS, std = CHANNEL_STDS)
    ])

TRANSFORM_MASK =  transforms.Compose(
    TRANSFORM_STRUCTURE + [
    # data transformation
    transforms.PILToTensor(),
])

# def transform_dual(instance):
#     image, target = instance
#     transform = RandomTransformsDual(TRANSFORM_STRUCTURE)
#     image, target = transform(image, target)

#     image = TRANSFORM_IMAGE(image)
#     target = TRANSFORM_MASK(target)

#     return image, target

def generate_data_loaders(args) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    trainset = Cityscapes(root = Path(args.data_path), split='train', mode='fine', 
                      transform = TRANSFORM_IMAGE, target_transform=TRANSFORM_MASK,
                      target_type='semantic')
    
    train_subset, val_subset = torch.utils.data.random_split(trainset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1))
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)
    wandb.log({"Data Loaded":dt.datetime.now()})
    
    
    return trainloader, valloader



    