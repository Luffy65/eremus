#!/usr/bin/env python
# coding: utf-8

import subprocess # Per istallare i package mancanti
import sys        # Per installare i package mancanti
import mne

# Function to install a package using pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import the torch module, and install it if not found
try:
    import torch
except ImportError:
    print("torch not found. Installing...")
    install('torch')
    import torch

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
# from math import isnan
from torch.utils.data import Dataset

class EremusDataset(Dataset):
    """A dataset for Emotion Recognition using EEG data and Musical Stimuli.
    
    Parameters
    ------------------
    xls_file : str
        Path to the xls file with samples annotations.
    eeg_root_dir : str
        Directory with all the eeg data.
    data_type : int 
        It describes the type of data to load:
        
        - EremusDataset.DATA_RAW (0) - data in *eeg_root_dir* are raw edf files; 
        - EremusDataset.DATA_PRUNED (1) - data in *eeg_root_dir* are pruned set files;
        - EremusDataset.DATA_PREPROCESSED (2) - data in *eeg_root_dir* are preprocessed npz files.
    indices : list of int
        Indices of xls_file to use in the current dataset. If None all samples are used.
    transform : callable 
        Optional transform to be applied on a sample.
    select_data : boolean
        If data_type != EremusDataset.DATA_PREPROCESSED and select_data = True, only channel data are selected, without metadata of Raw objects. Note that sample is a tuple of array: also time array is extracted. Do not use this attribute while using transforms.
    label_transform  : callable 
        Conversion function to be applied on a label.
    **args
        args to be passed to *label_transform*
    """
    
    DATA_RAW = 0
    """It describes raw edf data type."""
    DATA_PRUNED = 1
    """It describes set data type. EEG are filtered and pruned with ICA."""
    DATA_PREPROCESSED = 2
    """It describes npz data type. EEG are preprocessed through various techniques."""
    
    def __init__(self, xls_file, eeg_root_dir, data_type: int = DATA_RAW, indices=None, transform=None, select_data=False, label_transform=None, **args):
    
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.data_type = data_type
        self.indices = indices
        self.transform = transform
        self.select_data = select_data
        self.label_transform = label_transform
        self.args = args
        
        # if preprocessed data, load data in memory
        if self.data_type == 2:
            eeg_files = self.eeg_data.iloc[indices]['filename_preprocessed']
            # convert to set
            eeg_files = set(eeg_files)
            # load data
            self.data = {
                eeg_file: np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'] for eeg_file in tqdm(eeg_files)
            }
            

    def __len__(self):
        """
        It returns the len of the dataset. Lenght is retrieved from indices list or from xls file when the first is not present.
        """
        if self.indices is not None:
            return len(self.indices)
        return len(self.eeg_data)

    def __getitem__(self, idx):
        """
        It returns a single sample. It is invoked when you access the file with the brackets notazion (e.g. z_eremus[0]).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(idx)
        #print(self.eeg_data.iloc[idx])
        
        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]
        
        # get start and stop indices
        start = self.eeg_data.iloc[idx]['start_index']
        stop = self.eeg_data.iloc[idx]['end_index']
        
        #disable warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #get raw eeg
            if self.data_type == self.DATA_RAW:
                eeg_file = self.eeg_data.iloc[idx]['filename']
                raw_eeg = mne.io.read_raw_edf(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PRUNED:
                eeg_file = self.eeg_data.iloc[idx]['filename_pruned']
                raw_eeg = mne.io.read_raw_eeglab(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PREPROCESSED:
                eeg_file = self.eeg_data.iloc[idx]['filename_preprocessed']
                raw_eeg = self.data[eeg_file][:, start:stop]
                #raw_eeg = np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'][:, start:stop]
            else:
                raise Exception('Please provide correct data type')
        
        if self.data_type != self.DATA_PREPROCESSED:
            if self.select_data:
                #select only data --> no metadata info (no 'info object')
                raw_eeg = raw_eeg[:, start:stop]
            else:
                raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[start], tmax=raw_eeg.times[stop],  include_tmax=False)
        
        gew_emotion1 = eval(self.eeg_data.iloc[idx]['gew_1'])
        #gew_emotion2 = self.eeg_data.iloc[idx]['gew_2'] 
        
        #if isinstance(gew_emotion2, str):
        #    gew_emotion2 = eval(gew_emotion2)
        #elif isinstance(gew_emotion2, float):
        #    if isnan(gew_emotion2):
        #        gew_emotion2 = None
        
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = {
            'eeg': raw_eeg, 
            'emotion': emotion, 
            'subject_id': self.eeg_data.iloc[idx]['subject_id']
        }
        

        if self.transform:
            sample = self.transform(sample)

        return sample

EremusDataset_V2 = EremusDataset
