
from json import decoder
import torch
import torch.nn as nn
#from torch.func import functional_call, stack_module_state, vmap

""" segmentation model example
"""

class pre_trained_encoder(nn.Module):
    
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)
        
        return s1, s2, s3, s4, b
    
class standalone_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        
        """ Classifier """
        self.outputs = nn.Conv2d(64, 20, kernel_size=1, padding=0)
    
    def forward(self, s1, s2, s3, s4, b):
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Segmentation output """
        outputs = self.outputs(d4)

        return outputs      
    


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_rate=0.25):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        # self.dropout = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


# general network structure for ensamble model
class EnsambleModel(nn.Module):
    def __init__(self,encoder:pre_trained_encoder,decoders : list[standalone_decoder]):
        super().__init__()

        self.e = encoder
        self.decoders = torch.nn.ModuleList(decoders)
    #    self.params, self.buffers = stack_module_state(decoders)
          
    def freeze_encoder(self):
        for param in self.e.parameters():
            param.requires_grad = False  
    
    def forward(self, inputs):
        s1, s2, s3, s4, b = self.e(inputs)
        # outputs = vmap(self.fdecoder, in_dims=(0, 0, None))(self.params, self.buffers, s1, s2, s3, s4, b)
        outputs = [decoder(s1, s2, s3, s4, b) for decoder in self.decoders]
        return outputs
    
# function to populate this model for validation purposes 
def Model() -> EnsambleModel:
    
    def create_decoders(nr_decoders:int):
        decoders = []
        classes_to_ignore = []
        
        if 18 % nr_decoders != 0:
            raise Warning("Number of decoders must be a factor of 18")
        for i in range(nr_decoders):
            begin_class = i * int(18 / nr_decoders)
            end_class = (i + 1) * int(18 / nr_decoders)
            classes_to_ignore.append( list(range(begin_class,end_class)))
            
            decoder = standalone_decoder()
            decoders.append(decoder)
            
        return decoders
    
    encoder = pre_trained_encoder()
    decoders = create_decoders(3)
    return EnsambleModel(encoder,decoders)
    

    



