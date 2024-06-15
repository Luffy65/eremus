import torch.nn.functional as F
import torch
import scipy.io
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import numpy as np

class Model(nn.Module):
    """
    It defines a Convolutional Neural Network, SyncNet, adapted to EREMUS samples with extracted features in frequency-domain.
    Network is adapated for samples of size *(B, C, F, S)*, being *B* the batch size, *C* the number of EEG channels, *F* the number of extracted features and *S* the number of frequency bands.
    This particular implementation fits *S*=5. Other values for *S* were not tested.

    Arguments
    -------------
    args : dict
        A dictionary containing the following keys:
        
        num_channels : int
            The number of channels *C*. Default to 32.
        input_size : int
            The number of features *F*. Default to 19.
        num_classes : int
            The number of classes. Default to 4.
        num_filters : int
            Default to 1. Don't change it.
        filter_width : int
            Default to 40. Don't change it.
        pool_size : int
            Default to 40. Don't change it.
        verbose : bool
            If True, tensors sizes are printed at the end of each layer.
    
    See also
    --------------
    eremus.models.syncnet_eremus: a different version of SyncNet adapted for samples in time domain.
    """
    def __init__(self,args):
        super(Model, self).__init__()
        args_defaults=dict(num_channels=8, input_size=1266, num_classes=2, num_filters=1, filter_width = 40, pool_size=40, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        self.num_filters = self.num_filters 
        self.num_channels = self.num_channels
        self.filter_width = self.filter_width
        
        self.b = nn.Parameter(torch.FloatTensor(1,1,self.num_channels,self.num_filters).uniform_(-0.05, 0.05))
        self.bias = nn.Parameter(torch.FloatTensor(self.num_filters))
        self.omega = nn.Parameter(torch.FloatTensor(1,1,1,self.num_filters).uniform_(0, 1))
        self.cl_Wy = int(np.ceil(float(self.input_size)/float(self.pool_size)) * self.num_filters)
        
        if(self.filter_width%2 == 0):
            self.t = torch.FloatTensor(np.reshape(range(-int(self.filter_width/2),int(self.filter_width/2)),[1,int(self.filter_width),1,1]))
        else:
            self.t = torch.FloatTensor(np.reshape(range(-int((self.filter_width-1)/2),int((self.filter_width-1)/2) + 1),[1,int(self.filter_width) ,1,1]))
        self.t=nn.Parameter(self.t)
        self.phi_ini = nn.Parameter(torch.FloatTensor(1,1,self.num_channels, self.num_filters).normal_(0,0.05))
        self.beta = nn.Parameter(torch.FloatTensor(1,1,1,self.num_filters).uniform_(0, 0.05))
        
        ## Only stride and dilation values of 1 are supported. If you use different values, padding values wont be correct
        P = ((self.input_size-1)-self.input_size + (self.filter_width-1))
        if(P%2 == 0):
            self.padding = (P//2, P//2 + 1)
        else:
            self.padding = (P//2, P//2)
        print(self.padding)
        
        self.pool = nn.MaxPool2d((1, self.pool_size), stride = (1, self.pool_size))
        self.classifier = nn.Linear(self.input_size,self.num_classes)
    
    def forward(self,X):
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
        #X must be in the form of BxCx1xT or BxCxT
        self.W_osc = torch.mul(self.b,torch.cos(self.t*self.omega + self.phi_ini))
        self.W_decay = torch.exp(-torch.pow(self.t,2)*self.beta)
        self.W = torch.mul(self.W_osc,self.W_decay)
        self.W = self.W.view(self.num_filters,self.num_channels,1,self.filter_width)
        
        if self.verbose:
            print('[SYNCNET] Input after calculus: ', X.size())
        if(len(X.size()) == 3):
            X = X.unsqueeze(2)
        
        if self.verbose:
            print('[SYNCNET] Added dimension on pos 2 (if 3D):', X.size())
        res = F.conv2d(F.pad(X,self.padding,"constant", 0).float(),self.W.float(),bias = self.bias,stride=1 )
        if self.verbose:
            print('[SYNCNET] Convolutional layer: ', res.size())
        res = self.pool(F.pad(res,self.padding, "constant", 0))
        if self.verbose:
            print('[SYNCNET] Pooling: ', res.size())
        res = res.view(-1,self.input_size)
        if self.verbose:
            print('[SYNCNET] Collpase with view(-1): ', res.size())
        self.beta = nn.Parameter(torch.clamp(self.beta, min=0))
        if self.verbose:
            print('[SYNCNET] Parameter: ', res.size())
            
        res = F.relu(res)
        if self.verbose:
            print('[SYNCNET] Applied Relu', res.size())
            
        res = self.classifier(res)
        if self.verbose:
            print('[SYNCNET] Classifier output', res.size())
        
        #return self.classifier(res).squeeze() WHY USE SQUEEZE?
        return res