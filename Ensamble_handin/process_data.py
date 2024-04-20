import numpy as np
from torchvision import transforms
import torch
import logging

CHANNEL_MEANS = [0.485, 0.456, 0.406]
CHANNEL_STDS = [0.229, 0.224, 0.225]

IMG_SIZE = (512,1024)


TRANSFORM_STRUCTURE_VAL = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.LANCZOS)
])

TRANSFORM_IMAGE =  transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=CHANNEL_MEANS, std = CHANNEL_STDS)
    ])

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""

    img = TRANSFORM_STRUCTURE_VAL(img)
    img = TRANSFORM_IMAGE(img)
    img = img.unsqueeze(0)

    return img

def postprocess(mean_outputs, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    ensamble_output = torch.argmax(input=mean_outputs,dim=1)
    
    prediction = transforms.functional.resize(ensamble_output, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()
    
    # # predict ood
    # softmax_score_per_pixel, _ = torch.max(mean_outputs.permute(0,2,3,1), dim=3)
    # mean_softmax_score_of_image = torch.mean(softmax_score_per_pixel).item()
    
    # ood_treshold = 0.57
    # ood:bool = mean_softmax_score_of_image < ood_treshold
    
    # if ood:
    #     ood_counter.ood_images_count += 1
    #     logging.info(f"Percentage of OOD images: {ood_counter.percentage_ood()}")
    # else:
    #     ood_counter.id_images_count += 1
    
    return prediction_numpy

# for validation it is needed to know if the model is abler to flag for ood, due to the limited upload count for ood i have excluded activation scores that could be used for calibration.
# class ood_counter():
#     ood_images_count = 0
#     id_images_count = 0 
    
#     @classmethod
#     def percentage_ood(cls):
#         return cls.ood_images_count/(cls.ood_images_count+cls.id_images_count)
    
    
    








