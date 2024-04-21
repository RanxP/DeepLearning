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


# perspective transform
#rotation 10 % each direction

# color or gray scale transformations

def TRANSFORM_STRUCTURE(img, IMG_SIZE):
    # random.seed(torch.initial_seed())
    # img = transforms.RandomRotation(degrees=3)(img)

    # random.seed(torch.initial_seed())
    # # resize_factor =  random.uniform((IMG_SIZE[0]/1024),1 )
    # resize_factor =  random.uniform(0.8,1.2)
    # resized_img_size = (int(IMG_SIZE[0] * resize_factor), int(IMG_SIZE[1] * resize_factor))
    # img = transforms.RandomCrop(size=resized_img_size)(img)

    # manditory resize
    img = transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS)(img)

    return img

def TRANSFORM_STRUCTURE_VAL(img, IMG_SIZE):
    img = transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS)(img)
    return img

TRANSFORM_IMAGE =  transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=CHANNEL_MEANS, std = CHANNEL_STDS)
    ])

TRANSFORM_MASK =  transforms.Compose([
    transforms.PILToTensor(),
])

class RandomTransformsDual:
    def __init__(self, transform,IMG_SIZE):
        self.transform = transform
        self.IMG_SIZE = IMG_SIZE

    def __call__(self, img, target):
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed=seed)
        img = self.transform(img,self.IMG_SIZE)
        torch.manual_seed(seed=seed)
        target = self.transform(target,self.IMG_SIZE)
        #release the manual seed
        torch.seed()

        return img, target
    
class TransformDualInputCollection:
    def __init__(self, IMG_SIZE):
        self.IMG_SIZE = IMG_SIZE

    def transform_dual_train(self,image, target):
        transform = RandomTransformsDual(TRANSFORM_STRUCTURE,self.IMG_SIZE)
        image, target = transform(image, target)

        image = TRANSFORM_IMAGE(image)
        target = TRANSFORM_MASK(target)

        return image, target

    def transform_dual_val(self,image, target):
        image = TRANSFORM_STRUCTURE_VAL(image, self.IMG_SIZE)
        target = TRANSFORM_STRUCTURE_VAL(target, self.IMG_SIZE)
        image = TRANSFORM_IMAGE(image)
        target = TRANSFORM_MASK(target)

        return image, target

def generate_data_loaders(args) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    IMG_SIZE = [2**args.figure_size,2**(args.figure_size+1)]
    transform_collection = TransformDualInputCollection(IMG_SIZE)
    
    trainset = Cityscapes(root = Path(args.data_path), split='train', mode='fine', 
                    transforms=transform_collection.transform_dual_train, target_type='semantic')
    
    train_subset, _ = torch.utils.data.random_split(trainset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1480928))
    
    valset = Cityscapes(root = Path(args.data_path), split='train', mode='fine', 
                    transforms=transform_collection.transform_dual_val, target_type='semantic')
    
    _, val_subset = torch.utils.data.random_split(valset, [0.8, 0.2],
                                            generator=torch.Generator().manual_seed(1480928))
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            drop_last=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            drop_last=True)    
    
    return trainloader, valloader



    