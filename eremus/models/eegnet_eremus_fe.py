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
    Network is adapated for samples of size *(B, C, F)*, being *B* the batch size, *C* the number of EEG channels, and *F* the number of extracted features.
    This particular implementation fits only *F* = 32. If you want to customize the number of *F* you must play with some parameters through convolutional layers, as kernel size and stride. 
    We suggest to modify only the input size of the fully connected layer to fit your *F* size.

    Arguments
    -------------
    args : dict
        A dictionary containing the following keys:

        num_channels : int
            The number of channels. Default to 32.
        num_classes : int
            The number of classes. Default to 4.
        verbose : bool
            If True, tensors sizes are printed at the end of each layer.
    
    See also
    --------------
    eremus.models.eegnet_eremus: a different version of EEGNet adapted for samples in time domain.
    eremus.models.eegnet_eremus_fe_v2 : a different version of EEGNet adapted for samples in frequency domain.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults=dict(num_channels=182, num_classes=10, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, self.num_channels), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of features per sample in your data.
        # I have 32 features. 
        self.fc1 = nn.Linear(16, self.num_classes)
        

    def forward(self, x):
        """
        Parameters
        -------------
        x : torch.Tensor
            *x*  must be of size  *(B, C, F)*, being *B* the batch size, *C* the number of EEG channels and *F* the number of extracted features.
        
        Returns
        ----------
        torch.Tensor
            A tensor of size *(B, NC)*, being *B* the batch size, and *NC* the number of classes.
        """
        if self.verbose:
            print(x.size())#Initial size [B, C, F]
        x = x.unsqueeze(1)#Add additional'features dimension' [B, 1, C, F]
        x = x.transpose(2, 3)# [B, AF, F, C] AF=1

        # Layer 1
        if self.verbose:
            print(x.size())
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)# C dimension collides [B, 1, AF, F]
        
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


if __name__ == "__main__":
    net = Model(dict(num_channels=120, num_classes=10)).cuda(0)
    #print (net.forward(Variable(torch.Tensor(np.random.rand(16, 120, 1)).cuda(0))))