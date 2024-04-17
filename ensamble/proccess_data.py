import numpy as np
from torchvision import transforms
import torch

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

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    m = torch.nn.Softmax(dim=2)
    outputs_tensor = torch.stack(prediction) # shape (3,4,20,512,1024)
    normalized_outputs = m(outputs_tensor) # checked is correct
    mean_outputs = torch.mean(normalized_outputs, dim=0, keepdim=False)
    ensamble_output = torch.argmax(input=mean_outputs,dim=1)
    
    prediction = transforms.functional.resize(ensamble_output, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()
    
    # predict ood
    softmax_score_per_pixel, _ = torch.max(mean_outputs.permute(0,2,3,1), dim=3)
    mean_softmax_score_of_image = torch.mean(softmax_score_per_pixel).item()
    
    ood_treshold = 0.6
    ood:bool = mean_softmax_score_of_image < ood_treshold
    
    if ood:
        ood_counter.ood_images_count += 1
        print(f"Percentage of OOD images: {ood_counter.percentage_ood()}")
    else:
        ood_counter.id_images_count += 1
    
    return prediction_numpy

# for validation it is needed to know if the model works correctly for ood, due to the limmited upload count for ood i have excluded activation scores that could be used for calibration.
class ood_counter():
    ood_images_count = 0
    id_images_count = 0 
    
    def percentage_ood(self):
        return self.ood_images_count/(self.ood_images_count+self.id_images_count)
    
    
    








