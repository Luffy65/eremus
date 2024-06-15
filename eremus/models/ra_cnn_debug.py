import os
import sys
import math
import time
import random
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from eremus.utils.ADL import Adl_4d
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import torch; torch.utils.backcompat.broadcast_warning.enabled = True

class Model(nn.Module):
    """
    It implements the Regional Asymmetric Convolutional Neural Network.
    Network is adapated for samples of size *(B, 9, 9, T)*, being *B* the batch size, and *T* the number of time-points.
    Please use this implementation only for debug purposes, since, if *verbose* is True, lot of sizes information is shown. 
    It is particulary useful when you want the model fit your input data.
    This particular implementation accepts only *T* = 1280 time-points. If you want to customize the number of *T* you must play
    with some parameter through convolutional layers, as kernel size and stride. 
    We suggest to modify the kernel size of the last convolutional layer in Temporal Extractor to fit your *T* size.
   
    Arguments
    -------------
    args : dict
        A dictionary containing the following keys:
        
        num_channels : int
            The number of channels. Default to 32.
        num_classes : int
            The number of classes. Default to 4.
        verbose : bool
            If True, tensors sizes are printed at the end of each convolutional layer.
    
    See also
    --------------
    eremus.models.ra_cnn : the same version of RA-CNN with less verbose information
    eremus.models.ra_cnn_640 : a different version of RA-CNN adapted for 640 time-points
    eremus.models.spectral_ra_cnn : a different version of RA-CNN adapted for frequency-bands
    
    References
    ---------------
    Heng Cui, Aiping Liu, Xu Zhang, Xiang Chen, Kongqiao Wang, Xun Chen, 
    EEG-based emotion recognition using an end-to-end regional-asymmetric convolutional neural network,
    Knowledge-Based Systems,
    Volume 205,
    2020,
    106243,
    ISSN 0950-7051,
    https://doi.org/10.1016/j.knosys.2020.106243.
    """
    
    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults=dict(num_channels=32, num_classes=4, verbose=True)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        # TEMPORAL FEATURE EXTRACTOR
        # Layer 1
        self.conv1 = nn.Conv3d(1, 32, (1, 1, 5), stride=(1, 1, 2)) #padding = (0, 0, 2),
        self.batchnorm1 = nn.BatchNorm3d(32, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv3d(32, 32, (1, 1, 3), stride=(1, 1, 2)) #padding = (0, 0, 1),
        self.batchnorm2 = nn.BatchNorm3d(32, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv3d(32, 32, (1, 1, 3), stride=(1, 1, 2)) #padding = (0, 0, 1),
        self.batchnorm3 = nn.BatchNorm3d(32, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # Layer 4
        self.conv4 = nn.Conv3d(32, 32, (1, 1, 16), stride=(1, 1, 4))
        self.batchnorm4 = nn.BatchNorm3d(32, False)
        
        # Layer 5
        self.conv5 = nn.Conv3d(32, 32, (1, 1, 36))
        self.batchnorm5 = nn.BatchNorm3d(32, False)
        
        # REGIONAL FEATURE EXTRACTOR
        # Layer 1
        self.conv_reg1 = nn.Conv2d(32, 32, 3, padding=1)
        self.batchnorm_reg1 = nn.BatchNorm2d(32, False)
        
        # Layer 2
        self.conv_reg2 = nn.Conv2d(32, 32, 3, padding=1)
        self.batchnorm_reg2 = nn.BatchNorm2d(32, False)
        
        # ASYMMETRIC FEATURE EXTRACTOR
        # Layer 1
        self.conv_asym = nn.Conv2d(32, 64, 1, padding=0)
        self.batchnorm_asym = nn.BatchNorm2d(64, False)
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 1266 timepoints. 
        self.fc1 = nn.Linear(4896, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        

    def forward(self, x):
        """
        Parameters
        -------------
        x : torch.Tensor
            *x*  must be of size *(B, 9, 9, T)*, being *B* the batch size, and *T* the number of time-points.
        
        Returns
        ----------
        torch.Tensor
            A tensor of size *(B, NC)*, being *B* the batch size, and *NC* the number of classes.
        """
        # Add feature dimension 
        if self.verbose:
            print('[PREPARATION] Input size:', end='\t')
            print(x.size())
        x = x.unsqueeze(1)

        # Layer 1
        if self.verbose:
            print('\n[LAYER 1] Input size:', end='\t\t')
            print(x.size())
        x = F.elu(self.conv1(x))
        if self.verbose:
            print('[LAYER 1] Conv. Output size:', end='\t')
            print(x.size())
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        #x = x.permute(0, 3, 1, 2)
        if self.verbose:
            print('[LAYER 1] Batch N + Drop:', end='\t')
            print(x.size())
        
        # Layer 2
        if self.verbose:
            print('\n[LAYER 2] Input size:', end='\t\t')
            print(x.size())
        #x = self.padding1(x)
        if self.verbose:
            print('[LAYER 2] Padd. Output size:', end='\t')
            print(x.size(), end='NOT APPLIED\n')
        x = F.elu(self.conv2(x))
        if self.verbose:
            print('[LAYER 2] Conv. Output size:', end='\t')
            print(x.size())
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        #x = self.pooling2(x)
        if self.verbose:
            print('[LAYER 2] Pool. Output size:', end='\t')
            print(x.size(), end='NOT APPLIED\n')
        
        # Layer 3
        if self.verbose:
            print('\n[LAYER 3] Input size:', end='\t\t')
            print(x.size())
        #x = self.padding2(x)
        x = F.elu(self.conv3(x))
        if self.verbose:
            print('[LAYER 3] Conv. Outpute size:', end='\t')
            print(x.size())
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        #x = self.pooling3(x)
        if self.verbose:
            print('[LAYER 3] Pool. Output size:', end='\t')
            print(x.size(), end='NOT APPLIED\n')
        
        # Layer 4
        if self.verbose:
            print('\n[LAYER 4] Input size:', end='\t\t')
            print(x.size())
        #x = self.padding2(x)
        x = F.elu(self.conv4(x))
        if self.verbose:
            print('[LAYER 4] Conv. Outpute size:', end='\t')
            print(x.size())
        x = self.batchnorm4(x)
        x = F.dropout(x, 0.25)
        #x = self.pooling3(x)
        if self.verbose:
            print('[LAYER 4] Pool. Output size:', end='\t')
            print(x.size(), end='NOT APPLIED\n')
            
        # Layer 5
        if self.verbose:
            print('\n[LAYER 5] Input size:', end='\t\t')
            print(x.size())
        #x = self.padding2(x)
        x = F.elu(self.conv5(x))
        if self.verbose:
            print('[LAYER 5] Conv. Outpute size:', end='\t')
            print(x.size())
        x = self.batchnorm5(x)
        x = F.dropout(x, 0.25)
        #x = self.pooling3(x)
        if self.verbose:
            print('[LAYER 5] Pool. Output size:', end='\t')
            print(x.size(), end='NOT APPLIED\n')
        
        x.squeeze_(-1)
        # REGIONAL FEATURE EXTRACTOR
        # Layer 1
        if self.verbose:
            print('\n[LAYER R1] Input size:', end='\t\t')
            print(x.size())
        x0 = F.elu(self.conv_reg1(x))
        if self.verbose:
            print('[LAYER R1] Conv. Outpute size:', end='\t')
            print(x0.size())
        x0 = self.batchnorm_reg1(x0)
        x0 = F.dropout(x0, 0.25)
        
        # Layer 2
        if self.verbose:
            print('\n[LAYER R2] Input size:', end='\t\t')
            print(x0.size())
        x0 = F.elu(self.conv_reg2(x0))
        if self.verbose:
            print('[LAYER R2] Conv. Outpute size:', end='\t')
            print(x0.size())
        x0 = self.batchnorm_reg2(x0)
        x0 = F.dropout(x0, 0.25)
        
        # ASYMMETRIC FEATURE EXTRACTOR
        # Layer 1
        if self.verbose:
            print('\n[LAYER A1] Input size:', end='\t\t')
            print(x.size())
        x1 = Adl_4d(x)
        if self.verbose:
            print('[LAYER A1] ADL Output size:', end='\t')
            print(x1.size())
        x1 = F.elu(self.conv_asym(x1))
        if self.verbose:
            print('[LAYER A1] Conv. Outpute size:', end='\t')
            print(x1.size())
        x1 = self.batchnorm_asym(x1)
        x1 = F.dropout(x1, 0.25)

        
        # FC Layer
        if self.verbose:
            print('\n[LAYER FC] Input size 0:', end='\t')
            print(x0.size())
            print('[LAYER FC] Input size 1:', end='\t')
            print(x1.size())
        x0 = x0.view(x0.shape[0], -1)
        x1 = x1.view(x1.shape[0], -1)
        x = torch.cat((x0, x1), 1)
        if self.verbose:
            print('[LAYER FC] Dim collpase 0:', end='\t')
            print(x0.size())
            print('[LAYER FC] Dim collpase 1:', end='\t')
            print(x1.size())
            print('[LAYER FC] Features Fusion:', end='\t')
            print(x.size())
        x = torch.sigmoid(self.fc1(x))
        if self.verbose:
            print('[LAYER FC1] Output size:', end='\t')
            print(x.size())
        x = F.softmax(self.fc2(x), dim=1)
        if self.verbose:
            print('[LAYER FC2] Output size:', end='\t')
            print(x.size())
        return x

if __name__ == "__main__":
    net = Model(dict(num_channels=32, num_classes=22, verbose=True)).cuda(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)  