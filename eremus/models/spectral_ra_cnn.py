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
    It implements a frequency domain Regional Asymmetric Convolutional Neural Network.
    Network is adapated for samples of size *(B, 9, 9, F, S)*, being *B* the batch size, *F* the number of features, *S* the number of frequency bands.
    This particular implementation accepts only *S* = 5 frequency bands. If you want to customize the number of *S* you must play
    with some parameters through convolutional layers, as kernel size and stride, or you should add more layers.
   
    Arguments
    -------------
    args : dict
        A dictionary containing the following keys:
        
        input_size : int
            The size of input features. Default to 19.
        num_classes : int
            The number of classes. Default to 4.
        verbose : bool
            If True, tensors sizes are printed at the end of each convolutional layer.
    
    See also
    --------------
    eremus.models.ra_cnn : a different version of RA-CNN adapted for 1280 time-points
    eremus.models.ra_cnn_debug : a different version of RA-CNN with lot of debug information
    eremus.models.ra_cnn_640 : a different version of RA-CNN adapted for 640 time-points
    
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
        args_defaults=dict(input_size=19, num_classes=4, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        # SPECTRAL FEATURES EXTRACTOR
        self.spectral = nn.Sequential(
            
            # Layer 1
            # Set input size to number of features
            nn.Conv3d(self.input_size, 32, (1, 1, 2)),#, stride=(1, 1, 2)), #padding = (0, 0, 2),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25),
            
            # Layer 2
            #nn.ZeroPad2d((16, 17, 0, 1))
            nn.Conv3d(32, 32, (1, 1, 4)),#, stride=(1, 1, 2)), #padding = (0, 0, 1),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25),
            #nn.MaxPool2d(2, 4)
        )
        
        # REGIONAL FEATURES EXTRACTOR
        self.regional = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32, False),
            nn.Dropout(0.25),

            # Layer 2
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32, False),
            nn.Dropout(0.25)
        )
        
        # ASYMMETRIC FEATURES EXTRACTOR
        self.asymmetric = nn.Sequential(
            
            nn.Conv2d(32, 64, 1, padding=0),
            nn.ELU(),
            nn.BatchNorm2d(64, False),
            nn.Dropout(0.25)
        )
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 1280 timepoints. 
        self.fully_connected = nn.Sequential(
            
            nn.Linear(4896, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x):
        """
        Parameters
        -------------
        x : torch.Tensor
            *x*  must be of size *(B, 9, 9, F, S)*, being *B* the batch size, *F* the number of features, *S* the number of frequency bands.
        
        Returns
        ----------
        torch.Tensor
            A tensor of size *(B, NC)*, being *B* the batch size, and *NC* the number of classes.
        """
        # Use transpose [B, H, W, F, S] --> [B, F, H, W, S]
        x = x.permute(0, 3, 1, 2, 4)

        # Temporal Features Extractor
        if self.verbose:
            print('\n[SPECTRAL EXTRACTOR] Input size:', end='\t')
            print(x.size())
        x = self.spectral(x)
        if self.verbose:
            print('[SPECTRAL EXTRACTOR] Output size:', end='\t')
            print(x.size())
        
        # Depth is collapsed to 1: delete D dimension [B, F, H, W, 1] --> [B, F, H, W]
        x.squeeze_(-1)
        
        # Regional Features Extractor
        if self.verbose:
            print('\n[REGIONAL EXTRACTOR] Input size:', end='\t')
            print(x.size())
        x0 = self.regional(x)
        if self.verbose:
            print('[REGIONAL EXTRACTOR] Outpute size:', end='\t')
            print(x0.size())
        
        # Asymmetric Features Extractor
        if self.verbose:
            print('\n[ASYMMETRIC EXTRACTOR] Input size:', end='\t')
            print(x.size())
        x1 = self.asymmetric(Adl_4d(x))
        if self.verbose:
            print('[ASYMMETRIC EXTRACTOR] Output size:', end='\t')
            print(x1.size())

        
        # FC Layer
        # Collpase regional features, keep batch dimension [B, F, H, W] --> [B, F]
        x0 = x0.view(x0.shape[0], -1)
        # Collpase asymmetric features, keep batch dimension [B, F, H, W] --> [B, F]
        x1 = x1.view(x1.shape[0], -1)
        # Contatenate regional and asymmetric features along features dimension
        x = torch.cat((x0, x1), 1)
        if self.verbose:
            print('\n[FULLY CONNECTED LAYER] Input size:', end='\t')
            print(str(x0.size(1)) + '|' + str(x1.size(1)), end=' --> ')
            print(x.size())
        x = self.fully_connected(x)
        if self.verbose:
            print('[FULLY CONNECTED LAYER] Output size:', end='\t')
            print(x.size())
        return x