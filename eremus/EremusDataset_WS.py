import mne
import math
import torch
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class EremusDataset_WS(Dataset):
    """A dataset for Emotion Recognition using EEG data and MUsical Stimuli. When using this dataset class with a Sequential Sampler, access samples by shifting a time-window at each loading. A single counter is associated with the whole dataset. Then, you can see sliding window only when you have accessed ALL the samples. When the time window reaches the end for a sample, it resets to start for that sample. This is True only using a Sequential Sampler. It implements the sliding window single counter algorithm.
    
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
    windows_size : int
        The width of the time-window to use when accessing data. window size is expressed in number of sample (i.e. number of seconds * sampling frequency).
    step_size : int
        The number of sample (i.e. number of seconds * sampling frequency) with wich time-window is shifted through epoching.
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

    def __init__(self, xls_file, eeg_root_dir, data_type: int = DATA_RAW, window_size = 1280, step_size = 128*3, indices=None, transform=None, select_data=False, label_transform=None, **args):
    
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.data_type = data_type
        self.indices = indices
        self.transform = transform
        self.select_data = select_data
        self.label_transform = label_transform
        self.args = args
        
        self.window_size = window_size
        self.step_size = step_size
        self.count = 0
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]
        
        # get start and stop samples
        start = self.eeg_data.iloc[idx]['start_index']
        stop = self.eeg_data.iloc[idx]['end_index']
        sample_len = stop - start
        
        # calculate n_frames = number of frames per sample
        n_frames = math.floor((sample_len-self.window_size)/self.step_size) + 1   
        
        # window selection
        window_start = start + math.floor(self.count/len(self))%n_frames * self.step_size
        window_stop = window_start + self.window_size
        #print(self.count, window_start - start, window_stop - start)
        
        # update count
        self.count = self.count + 1
        
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
                raw_eeg = np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'][:, window_start:window_stop]
            else:
                raise Exception('Please provide correct data type')
        
        if self.data_type != self.DATA_PREPROCESSED:
            if self.select_data:
                #select only data --> no metadata info (no 'info object')
                raw_eeg = raw_eeg[:, window_start:window_stop]
            else:
                raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[window_start], tmax=raw_eeg.times[window_stop],  include_tmax=False)
        
        gew_emotion1 = eval(self.eeg_data.iloc[idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = {'eeg': raw_eeg, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class EremusDataset_WS_MultipleCounters(Dataset):
    """A dataset for Emotion Recognition using EEG data and MUsical Stimuli. When using this dataset class, we access samples by shifting a time-window at each sample loading. A counter is associated with each sample. Then you can see sliding window just with two consecutives accesses of the same sample. When the time window reaches the end for a sample, it resets to start for that sample. It implements the sliding window multiple counters algorithm.
    
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
    windows_size : int
        The width of the time-window to use when accessing data. window size is expressed in number of sample (i.e. number of seconds * sampling frequency).
    step_size : int
        The number of sample (i.e. number of seconds * sampling frequency) with wich time-window is shifted through epoching.
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


    def __init__(self, xls_file, eeg_root_dir, data_type: int = DATA_RAW, window_size = 1280, step_size = 128*3, indices=None, transform=None, select_data=False, label_transform=None, **args):
    
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.data_type = data_type
        self.indices = indices
        self.transform = transform
        self.select_data = select_data
        self.label_transform = label_transform
        self.args = args
        
        self.window_size = window_size
        self.step_size = step_size
        self.counters = None
        if self.indices is not None:
            self.counters = list(np.zeros(len(self.indices), dtype=np.int))
        else: 
            self.counters = list(np.zeros(len(self.eeg_data), dtype=np.int))
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            real_idx = self.indices[idx]
            count = self.counters[idx]
        else: 
            real_idx = idx
            count = self.counters[idx]
        
        # get start and stop samples
        start = self.eeg_data.iloc[real_idx]['start_index']
        stop = self.eeg_data.iloc[real_idx]['end_index']
        sample_len = stop - start
        
        # calculate n_frames = number of frames per sample
        n_frames = math.floor((sample_len-self.window_size)/self.step_size) + 1   
        
        # window selection
        window_start = start + math.floor(count)%n_frames * self.step_size
        window_stop = window_start + self.window_size
        #print(count, window_start - start, window_stop - start)
        
        # update count
        if self.counters is not None:
            self.counters[idx] = count + 1
            
        #disable warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #get raw eeg
            if self.data_type == self.DATA_RAW:
                eeg_file = self.eeg_data.iloc[real_idx]['filename']
                raw_eeg = mne.io.read_raw_edf(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PRUNED:
                eeg_file = self.eeg_data.iloc[real_idx]['filename_pruned']
                raw_eeg = mne.io.read_raw_eeglab(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PREPROCESSED:
                eeg_file = self.eeg_data.iloc[real_idx]['filename_preprocessed']
                raw_eeg = np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'][:, window_start:window_stop]
            else:
                raise Exception('Please provide correct data type')
        
        if self.data_type != self.DATA_PREPROCESSED:
            if self.select_data:
                #select only data --> no metadata info (no 'info object')
                raw_eeg = raw_eeg[:, window_start:window_stop]
            else:
                raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[window_start], tmax=raw_eeg.times[window_stop],  include_tmax=False)
        
        gew_emotion1 = eval(self.eeg_data.iloc[real_idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = {'eeg': raw_eeg, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    
    import splitter
    from gew import gew_to_hldv4
    from torchvision import transforms
    from eeg_transforms import *

    # split dataset into train, validation and test
    train_idx, validation_idx, test_idx = splitter.simple_split('eremus_test.xlsx', 
                                                                train_frac=0.7, 
                                                                validation_frac=0.15, 
                                                                test_frac=0.15)

    # define transforms
    crop = RandomCrop(512)
    to_matrix = ToMatrix()
    to_tensor = ToTensor(interface='unpacked_values', label_interface='long')

    # compose transforms
    composed = transforms.Compose([crop, to_matrix, to_tensor])

    emus_train = EremusDataset_WS_MultipleCounters(xls_file='eremus_test.xlsx', 
                                  eeg_root_dir='preprocessed\\overall_z_score_base\\', 
                                  data_type=EremusDataset_WS_MultipleCounters.DATA_PREPROCESSED,  
                                  window_size=1280, 
                                  step_size=128*3,
                                  indices = train_idx, 
                                  transform=composed, 
                                  label_transform=gew_to_hldv4)

    emus_validation = EremusDataset_WS_MultipleCounters(xls_file='eremus_test.xlsx', 
                                  eeg_root_dir='preprocessed\\overall_z_score_base\\',
                                  data_type=EremusDataset_WS_MultipleCounters.DATA_PREPROCESSED,  
                                  window_size=1280, 
                                  step_size=128*3,
                                  indices = validation_idx, 
                                  transform=composed, 
                                  label_transform=gew_to_hldv4)

    emus_test = EremusDataset_WS_MultipleCounters(xls_file='eremus_test.xlsx', 
                                  eeg_root_dir='preprocessed\\overall_z_score_base\\', 
                                  data_type=EremusDataset_WS_MultipleCounters.DATA_PREPROCESSED,  
                                  window_size=1280, 
                                  step_size=128*3,
                                  indices = test_idx, 
                                  transform=composed, 
                                  label_transform=gew_to_hldv4)

    print("Train dataset: " + str(len(emus_train)) + "\tValidation dataset: " + str(len(emus_validation)) + "\tTest dataset: " + str(len(emus_test)))

