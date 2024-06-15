# EremusDataset - ExtractedFeatures

import mne
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class EremusDataset_EF(Dataset):
    """A dataset for Emotion Recognition using EEG data and MUsical Stimuli. Use this dataset if you have preprocessed sample with shape (C, F, B), being *C* the number of channels, *F* the number of features and *B* the number of frequency bands. Each sample must be saved in a npz file, with index as its name (i.e. 100.nzp for 100th sample). npz file must contain a single array.
    
    Parameters
    ------------------
    xls_file : str
        Path to the xls file with samples annotations.
    root_dir : str
        Directory with all the preprocessed data.
    indices : list of int
        Indices of xls_file to use in the current dataset. If None all samples are used.
    transform : callable 
        Optional transform to be applied on a sample.
    label_transform  : callable 
        Conversion function to be applied on a label.
    **args
        args to be passed to *label_transform*
    """
    
    def __init__(self, xls_file, root_dir, indices=None, transform=None, label_transform=None, **args):
        
        self.data = pd.read_excel(xls_file)        
        self.root_dir = root_dir
        self.indices = indices
        self.transform = transform
        self.label_transform = label_transform
        self.args = args
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]

        #get extracted feature
        features = np.load(self.root_dir + str(idx) + '.npz')['arr_0']
        
        gew_emotion1 = eval(self.data.iloc[idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = features, emotion

        if self.transform:
            sample = self.transform(sample)

        return sample

class EremusDataset_EFWS(Dataset):
    """A dataset for Emotion Recognition using EEG data and MUsical Stimuli. Use this dataset if you have preprocessed sample with shape (C, F, B), being *C* the number of channels, *F* the number of features and *B* the number of frequency bands. Each sample must be saved in a npz file, with index as its name (i.e. 100.nzp for 100th sample). npz file must contain n_frames arrays, one per time-window, based on window size and step size parameters.
    
    Parameters
    ------------------
    xls_file : str
        Path to the xls file with samples annotations.
    root_dir : str
        Directory with all the preprocessed data.
    windows_size : int
        The width of the time-window to use when accessing data. window size is expressed in number of sample (i.e. number of seconds * sampling frequency).
    step_size : int
        The number of sample (i.e. number of seconds * sampling frequency) with wich time-window is shifted through epoching.
    indices : list of int
        Indices of xls_file to use in the current dataset. If None all samples are used.
    transform : callable 
        Optional transform to be applied on a sample.
    label_transform  : callable 
        Conversion function to be applied on a label.
    **args
        args to be passed to *label_transform*
    """

    def __init__(self, xls_file, root_dir, indices=None, transform=None, window_size = 1280, step_size = 128*3, label_transform=None, **args):
    
        self.data = pd.read_excel(xls_file)        
        self.root_dir = root_dir
        self.indices = indices
        self.transform = transform
        self.label_transform = label_transform
        self.args = args
        
        self.window_size = window_size
        self.step_size = step_size
        
        # create index map
        self.index_dict = {}
        index_count = 0
        if indices is not None:
            self.data_selection = self.data.iloc[indices]
        else:
            self.data_selection = self.data
        for i, row in self.data_selection.iterrows():
            # compute sample length
            sample_len = row.end_index - row.start_index
            # calculate n_frames = number of frames per sample
            n_frames = math.floor((sample_len-self.window_size)/self.step_size) + 1
            # map index count to (i, n_frame)
            for n_frame in range(n_frames):
                self.index_dict[index_count] = (i, n_frame)
                index_count = index_count + 1
        
    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        #if self.indices is not None:
        #    idx = self.indices[idx]
        
        # get sample id and offset
        sample_id, offset = self.index_dict[idx]
        file = str(sample_id) + '.npz'
        features = np.load(self.root_dir + file)['arr_' + str(offset)] # use offset in the correct way

        gew_emotion1 = eval(self.data.iloc[sample_id]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = features, emotion

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":

    
    import random
    from gew import gew_to_hldv5
    from features_transforms import *
    from torchvision import transforms
    
    #=============
    # Testing EF
    #=============


    # list all indices
    emus_dataset = pd.read_excel('eremus_test.xlsx')
    indices = list(range(len(emus_dataset)))

    # calculate different and neutral emotions' indices
    def select_emotion(labels, emotion_to_delete):    
        return [i for i, x in enumerate(labels) if x == emotion_to_delete]

    labels = [eval(row.gew_1)[0] for _, row in emus_dataset.iterrows()]
    different_emotions = select_emotion(labels, 21)
    neutral_emotions = select_emotion(labels, 20)

    # exclude different emotions
    indices = [i for i in indices if i not in different_emotions]

    # split into train, validation and test
    random.shuffle(indices)

    train_fraction = 0.7
    validation_fraction = 0.15
    test_fraction = 0.15

    num_test = int(len(indices)*test_fraction)
    num_validation = int(len(indices)*validation_fraction)
    num_train = len(indices) - num_test - num_validation

    train_idx = indices[0:num_train]
    validation_idx = indices[num_train:num_train+num_validation]
    test_idx = indices[num_train+num_validation:]

    # print infos
    print("Train samples: " + str(num_train) + "\tValidation samples: " + str(num_validation) + "\tTest samples: " + str(num_test))

    # load train dataset
    emus_train = EremusDataset_EF(xls_file='eremus_test.xlsx', 
                                  root_dir='preprocessed\\de\\', 
                                  indices=train_idx, 
                                  label_transform=gew_to_hldv5)

    #-------------
    # Transforms
    #-------------

    # Define transforms
    crop = RandomCrop(200)
    gamma_band = SelectBand('gamma')
    to_matrix = ToMatrix()
    to_tensor = ToTensor()

    composed = transforms.Compose([crop, gamma_band, to_matrix, to_tensor])

    # load test dataset
    emus_test = EremusDataset_EF(xls_file='eremus_test.xlsx', 
                                 root_dir='preprocessed\\de\\', 
                                 indices=test_idx, 
                                 transform = composed,
                                 label_transform=gew_to_hldv5)

    emus_test[0][0].shape


    #===============
    # Testing EFWS
    #===============

    import random
    from gew import gew_to_hldv5
    WINDOW_SIZE = 10
    STEP_SIZE = 3
    SFREQ = 128
    DIR = str(WINDOW_SIZE) + '_' + str(STEP_SIZE) + '\\'

    # list all indices
    emus_dataset = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                      root_dir='preprocessed\\de_ws\\'+DIR, 
                                      window_size=WINDOW_SIZE*SFREQ, 
                                      step_size=STEP_SIZE*SFREQ)
    indices = list(range(len(emus_dataset)))

    # split into train, validation and test
    random.shuffle(indices)

    train_fraction = 0.7
    validation_fraction = 0.15
    test_fraction = 0.15

    num_test = int(len(indices)*test_fraction)
    num_validation = int(len(indices)*validation_fraction)
    num_train = len(indices) - num_test - num_validation

    train_idx = indices[0:num_train]
    validation_idx = indices[num_train:num_train+num_validation]
    test_idx = indices[num_train+num_validation:]

    # print infos
    print("Train samples: " + str(num_train) + "\tValidation samples: " + str(num_validation) + "\tTest samples: " + str(num_test))

    # load train dataset
    emus_train = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                      root_dir='preprocessed\\de_ws\\'+DIR, 
                                      window_size=WINDOW_SIZE*SFREQ, 
                                      step_size=STEP_SIZE*SFREQ, 
                                      indices=train_idx)

    #-------------
    # Transforms
    #-------------

    # Define transforms
    crop = FixedCrop(32)
    gamma_band = SelectBand('theta')
    to_matrix = ToMatrix()
    to_tensor = ToTensor()

    composed = transforms.Compose([crop, gamma_band, to_matrix, to_tensor])

    # load test dataset
    emus_val = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                  root_dir='preprocessed\\de_ws\\'+DIR, 
                                  window_size=WINDOW_SIZE*SFREQ, 
                                  step_size=STEP_SIZE*SFREQ, 
                                  indices=train_idx,
                                  transform=composed)
    # Expected 9x9x32
    emus_val[0][0].shape


    # Define transforms
    crop = RandomCrop(32)
    gamma_band = SelectBand('beta')
    to_tensor = ToTensor()

    composed = transforms.Compose([crop, gamma_band, to_tensor])

    # load test dataset
    emus_val = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                  root_dir='preprocessed\\de_ws\\'+DIR, 
                                  window_size=WINDOW_SIZE*SFREQ, 
                                  step_size=STEP_SIZE*SFREQ, 
                                  indices=train_idx,
                                  transform=composed)
    # Expected 32x32
    emus_val[0][0].shape

