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
import datetime as dt
import random

CHANNEL_MEANS = [0.485, 0.456, 0.406]
CHANNEL_STDS = [0.229, 0.224, 0.225]

IMG_SIZE = (512,1024)

# ideas 5 crop
# perspective transform
#rotation 10 % each direction

# color or gray scale transformations

def TRANSFORM_STRUCTURE(img):
    random.seed(torch.initial_seed())
    img = transforms.RandomRotation(degrees=3)(img)

    random.seed(torch.initial_seed())
    # resize_factor =  random.uniform((IMG_SIZE[0]/1024),1 )
    resize_factor =  random.uniform(0.8,1.2)
    resized_img_size = (int(IMG_SIZE[0] * resize_factor), int(IMG_SIZE[1] * resize_factor))
    img = transforms.RandomCrop(size=resized_img_size)(img)

    img = transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS)(img)

    return img

TRANSFORM_STRUCTURE_VAL = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS)
])

TRANSFORM_IMAGE =  transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=CHANNEL_MEANS, std = CHANNEL_STDS)
    ])

TRANSFORM_MASK =  transforms.Compose([
    transforms.PILToTensor(),
])

class RandomTransformsDual:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target):
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed=seed)
        img = self.transform(img)
        torch.manual_seed(seed=seed)
        target = self.transform(target)
        #release the manual seed
        torch.seed()

        return img, target

def transform_dual_train(image, target):
    # image, target = instance
    transform = RandomTransformsDual(TRANSFORM_STRUCTURE)
    image, target = transform(image, target)

    image = TRANSFORM_IMAGE(image)
    target = TRANSFORM_MASK(target)

    return image, target

def transform_dual_val(image, target):
    image = TRANSFORM_STRUCTURE_VAL(image)
    target = TRANSFORM_STRUCTURE_VAL(target)
    image = TRANSFORM_IMAGE(image)
    target = TRANSFORM_MASK(target)

    return image, target

def generate_data_loaders(args) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    trainset = Cityscapes(root = Path(args.data_path), split='train', mode='fine', 
                    transforms=transform_dual_train, target_type='semantic')
    
    train_subset, _ = torch.utils.data.random_split(trainset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1))
    
    valset = Cityscapes(root = Path(args.data_path), split='train', mode='fine', 
                    transforms=transform_dual_val, target_type='semantic')
    
    _, val_subset = torch.utils.data.random_split(valset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1))
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            drop_last=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            drop_last=True)    
    
    return trainloader, valloader



    