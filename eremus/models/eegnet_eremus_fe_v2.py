import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
    
class Model(nn.Module):
    """
    It defines a Convolutional Neural Network, EEGNet, adapted to EREMUS samples with extracted features in frequency-domain.
    Network is adapated for samples of size *(B, C, F, S)*, being *B* the batch size, *C* the number of EEG channels, *F* the number of extracted features and *S* the number of frequency bands.
    This particular implementation fits only *F* = 19 and *S*=5. If you want to customize the number of *F* you must set it into *args* and play with some parameters through convolutional layers, as kernel size and stride. 
    We suggest to modify only the input size of the fully connected layer to fit your *F* size.

    Arguments
    -------------
    args : dict
        A dictionary containing the following keys:
        
        input_size : int
            The size of input features. Default to 19.
        num_channels : int
            The number of channels. Default to 32.
        num_classes : int
            The number of classes. Default to 4.
        verbose : bool
            If True, tensors sizes are printed at the end of each layer.
    
    See also
    --------------
    eremus.models.eegnet_eremus: a different version of EEGNet adapted for samples in time domain.
    eremus.models.eegnet_eremus_fe : a different version of EEGNet adapted for samples with extracted features.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults=dict(input_size=19, num_channels=32, num_classes=4, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        # Layer 1
        self.conv1 = nn.Conv2d(self.input_size, 32, (1, self.num_channels), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(32, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 5))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of features per sample in your data.
        self.fc1 = nn.Linear(32, self.num_classes)
        

    def forward(self, x):
        """
        Parameters
        -------------
        x : torch.Tensor
            *x*  must be of size  *(B, C, F, S)*, being *B* the batch size, *C* the number of EEG channels, *F* the number of extracted features and *S* the number of frequency bands.
        
        Returns
        ----------
        torch.Tensor
            A tensor of size *(B, NC)*, being *B* the batch size, and *NC* the number of classes.
        """
        if self.verbose:
            print(x.size())
        # [B, C, F, S] --> [B, F, S, C]
        x = x.permute(0, 2, 3, 1)

        # Layer 1
        if self.verbose:
            print(x.size())
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        # [B, F, S, C] --> [B, C, F, S]
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        if self.verbose:
            print(x.size())
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        if self.verbose:
            print(x.size())
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        if self.verbose:
            print(x.size())
        x = x.view(x.shape[0], -1)
        if self.verbose:
            print(x.size())
        x = torch.sigmoid(self.fc1(x))
        if self.verbose:
            print(x.size())
        return x

net = Model(dict(num_channels=32, num_classes=22, verbose=True)).cuda(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(dev)