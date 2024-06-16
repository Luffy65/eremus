import subprocess
import sys

# Function to install a package using pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import mne
import math
import json
import torch
import warnings
import numpy as np
import pandas as pd
from math import isnan
from typing import Union

# Try to import the torchvision module, and install it if not found
try:
    from torchvision import transforms, utils
except ImportError:
    print("torchvision not found. Installing...")
    install('torchvision')
    from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader

class EremusDataset_SingleSubject_FE(Dataset):
    """A dataset for Emotion Recognition using EEG data and MUsical Stimuli. Use this dataset if you have single-subject spreadsheet. Please note that, comparared with other dataset classes, this one is simpler, but it requires more complexity one spreadsheet creation: *indices* parameter does not exist, so you have to create a xls file with the only samples you need. Moreover all sampler must be preprocessed sample with shape (C, F, B), being *C* the number of channels, *F* the number of features and *B* the number of frequency bands. Each sample must be saved in a npz file, with is orignal index as its name (i.e. 100.nzp for 100th sample). Single pz file could contain multiple arrays, (for example with sliding window). The, for each sample, you must indicate in xls file also the *array_index* parameter. Examples of correct spreadsheets are produced by splitter.de_ws_temporal_split, splitter.de_ws_rnd_temporal_split, splitter.de_ws_kfold_temporal_split or splitter.de_ws_simple_split.
    
    Parameters
    ------------------
    xls_file : str
        Path to the xls file with samples annotations.
    root_dir : str
        Directory with all the preprocessed data.
    transform : callable 
        Optional transform to be applied on a sample.
    label_transform  : callable 
        Conversion function to be applied on a label.
    **args
        args to be passed to *label_transform*
        
    See Also
    -------------------
    splitter.de_ws_temporal_split, splitter.de_ws_rnd_temporal_split, splitter.de_ws_kfold_temporal_split, splitter.de_ws_simple_split: for creating a valid xls file for this dataset.
    """
    
    def __init__(self, xls_file, root_dir, transform=None, label_transform=None, **args):
        
        self.data = pd.read_excel(xls_file)        
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform
        self.args = args
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get original and array indices
        original_index = self.data.iloc[idx].original_index
        array_index = self.data.iloc[idx].array_index

        #get extracted feature
        features = np.load(self.root_dir + str(original_index) + '.npz')['arr_'+str(array_index)]
        
        gew_emotion1 = eval(self.data.iloc[idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = features, emotion

        if self.transform:
            sample = self.transform(sample)

        return sample