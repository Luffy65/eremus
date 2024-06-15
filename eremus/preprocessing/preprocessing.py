"""
In this module we are focusing on preprocessing steps: normalization and standardization.

Tips
--------------
Execute normalization or standardization while in numpy format, before convert to tensors.
Most of steps used in this impelementation are already avaliable in numpy library and not in torch.
"""
import os
import mne
import json
import torch
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

# configure eremus Path
configuration_path = Path(__file__).parent.parent
with open(configuration_path / 'configuration.txt') as json_file:
    configuration = json.load(json_file)
    path_to_eremus_data = Path(configuration['path_to_eremus_data'])
dataset_file = pd.read_excel(path_to_eremus_data/'eremus_test.xlsx')
pruned_eeg_root_dir = path_to_eremus_data/Path('recordings_pruned_with_ICA')
preprocessed_eeg_root_dir = path_to_eremus_data/Path('preprocessed_data')

class ResidualNan(Exception):
    """Describers the presence of NaNs in data"""
    pass

def interpolate(raw_data):
    """
    Replace nans values with linear interpolation under the hypotesis that errors are always punctual.
    
    Parameters
    -------------
    raw_data : numpy.ndarray
        The raw data. 
    
    Returns
    ------------
    numpy.ndarray
        the pruned data into the input's format, without nans
    """
    # get indices of nans
    nan_indices = np.where(np.isnan(raw_data))
    nan_indices = np.vstack(nan_indices).transpose()

    # hypotesis, Punctual nans
    for channel, timepoint in nan_indices:

        # get value before the point
        before = raw_data[channel, timepoint-1]
        # get value after the point
        after = raw_data[channel, timepoint-1]

        # interpolate
        raw_data[channel, timepoint] = (before + after) / 2

    nan_indices = np.where(np.isnan(raw_data))
    nan_indices = np.vstack(nan_indices).transpose()
    any_nan = nan_indices.shape[0]!=0
    if any_nan:
        raise ResidualNan("Data still contain Nans after interpolation")
        
    return raw_data

def z_score_norm(raw_eeg_t, mean=None, std=None):
    """
    Change data distribution to meet standardization constraints (0-mean, 1-std).
    Mean and standard deviation are calculated above all channels, from the single eeg sample (i.e. outputs single mean and std)
    
    Parameters
    -------------
    raw_eeg_t : Union[torch.Tensor, numpy.ndarray]
        The raw data. 
    mean : float 
        The mean value used in z-score computation. if None mean is calculated over the input.
    std : float 
        The standard deviation value used in z-score computation. if None std is calculated over the input.
    
    Returns
    ------------
    Union[torch.Tensor, numpy.ndarray]
        the standardized data into the input's format.
    """
    if (mean is None) != (std is None):
        raise Exception ('Please provide both or neither, mean and std args')
        
    if mean is None and std is None:
        mean = raw_eeg_t.mean()
        std = raw_eeg_t.std()

    return ((raw_eeg_t - mean)/std)

def minmax_norm(raw_eeg_t, a, b, minim=None, maxim=None):
    """
    Change data distribution to meet normalization constraints (data in range [a, b]).
    Maximum and minimum values are calculated above all channels, from the single eeg sample (i.e. outputs single max and min)
    
    Parameters
    -------------
    raw_eeg_t : Union[torch.Tensor, numpy.ndarray]
        The raw data. 
    a : float
        The mininum value in the normalized output.
    b : float
        The maximum value in the normalized output.
    minim : float
        The minim value used in minmax computation. if None minimum is calculated over the input.
    maxim : float
        The maximum value used in minmax computation. if None maximum is calculated over the input.
    
    Returns
    ------------
    Union[torch.Tensor, numpy.ndarray]
        the normalized data into the input's format.
    """
    if (minim is None) != (maxim is None):
        raise Exception ('Please provide both or neither, minim and maxim args')
        
    if minim is None and maxim is None:
            minim = raw_eeg_t.min()
            maxim = raw_eeg_t.max()
    w = b - a
    
    return (w*(raw_eeg_t - minim)/(maxim - minim) + a)

# Along axes (mean and std for dimensions)
def channel_z_score_norm(raw_eeg_t, mean=None, std=None):
    """
    Change data distribution to meet standardization constraints (0-mean, 1-std).
    Mean and standard deviation are calculated for each channel (i.e. outputs multiple means and stds (one for channel))
    
    Parameters
    -------------
    raw_eeg_t : Union[torch.Tensor, numpy.ndarray]
        The raw data. 
    mean : float 
        The mean value used in z-score computation. if None mean is calculated over the input.
    std : float 
        The standard deviation value used in z-score computation. if None std is calculated over the input.
    
    Returns
    ------------
    Union[torch.Tensor, numpy.ndarray]
        the standardized data into the input's format.
    """
    if (mean is None) != (std is None):
        raise Exception ('Please provide both or neither, mean and std args')
        
    if mean is None and std is None:
        mean = raw_eeg_t.mean(1)
        std = raw_eeg_t.std(1)

    return ((raw_eeg_t.transpose(1, 0) - mean)/std).transpose(1, 0)

def channel_minmax_norm(raw_eeg_t, a, b, minim=None, maxim=None):
    """
    Change data distribution to meet normalization constraints (data in range [a, b]).
    Maximum and minimum values are calculated for each channel (i.e. outputs multiple maxs and mins (one for channel))
    
    Parameters
    -------------
    raw_eeg_t : Union[torch.Tensor, numpy.ndarray]
        The raw data. 
    a : float
        The mininum value in the normalized output.
    b : float
        The maximum value in the normalized output.
    minim : Union[torch.Tensor, numpy.ndarray]
        The minim value used in minmax computation. if None minimum is calculated over the input.
    maxim : Union[torch.Tensor, numpy.ndarray]
        The maximum value used in minmax computation. if None maximum is calculated over the input.
    
    Returns
    ------------
    Union[torch.Tensor, numpy.ndarray]
        the normalized data into the input's format.
    """
    if (minim is None) != (maxim is None):
        raise Exception ('Please provide both or neither, minim and maxim args')
        
    if minim is None and maxim is None:
        if isinstance(raw_eeg_t, np.ndarray):
            minim = raw_eeg_t.min(1)
            maxim = raw_eeg_t.max(1)
        else:
            minim, _ = raw_eeg_t.min(1)
            maxim, _ = raw_eeg_t.max(1)
    w = b - a
    
    return (w*(raw_eeg_t.transpose(1, 0) - minim)/(maxim - minim) + a).transpose(1, 0)

# ## Generalize Normalization and Standardization
# 
# Normalization and standardization could be easily calculated in place using sample statistics, but not including other samples.
# In order to generalize normalization and standardization, we have to precompute means, stds, mins and maxs for each sample and then aggregate informations to produce a table with these entries for each sample:
# - stats over the sample (the whole - not cropped - song)
# - stats over the subject (using all samples of a single - or both, personal and other - recording(s))
# - stats over the subject (using recorded baseline)
# - stats over the dataset (using all samples of all recordings)
# - stats over the dataset (using recorded baseline of all recordings)

# In order to calculate stats over subjects and over dataset, we need a mean to aggregate computed stats:

# mean of means
def mean_of_means(means, weights):
    """
    Given means and lenghts of N not-overlapping subsets of a whole set, compute the mean of the set. Weights are the subsets'lengths.
    
    Parameters
    -----------
    means : Union[list, numpy.ndarray]
        The list of means, one for each subset. The single mean could be a single value (calculated over the whole subset) or a numpy.ndarray (a per-channel mean).
    weights : Union[list, numpy.ndarray]
        The list of lenghts, one for each subset.
        
    Returns
    -----------
    Union[numpy.float64, numpy.ndarray]
        The mean of the whole set (union of subsets). The returned mean could be a single value (calculated over the whole set) or a numpy.ndarray (calculated for each channel), depending on the shape of means. 
    int
        The sum of the weights (that will be the weight of the set).
    """
    means = np.array(means)
    if len(means[0].shape)==1:
        # means per channel
        #means = np.array([np.array(t) for t in stats['mean']])
        return np.average(means, 0, weights=np.array(weights)), sum(weights)
    else:
        # one mean value for all channels
        return np.average(means, weights=np.array(weights)), sum(weights)

# std of stdts
def std_of_stds(stds, means, weights):
    """
    Given stds, means and lenghts of N not-overlapping subsets of a whole set, compute the standard deviation of the set. Weights are the subsets'lengths.
    
    Parameters
    ------------
    stds : Union[list, numpy.ndarray]
        The list of stds, one for each subset. The single mean could be a single value (calculated over the whole subset) or a or a numpy.ndarray (calculated for each channel).
    means : Union[list, numpy.ndarray]
        The list of means, one for each subset. The single mean could be a single value (calculated over the whole subset) or a numpy.ndarray (a per-channel mean).
    weights : Union[list, numpy.ndarray]
        The list of lenghts, one for each subset.
        
    Returns
    -----------
    Union[numpy.float64, numpy.ndarray]
        The standard deviation of the whole set (union of subsets).
        The returned standard deviation could be a single value (calculated over the whole set) or a numpy.ndarray (calculated for each channel), depending on the shape of stds.
    """
    stds = np.array(stds)
    means = np.array(means)
    weights = np.array(weights)
    
    ov_mean, weights_sum = mean_of_means(means, weights)
    dev = means - ov_mean
    
    if len(means[0].shape)==1:
        # means per channel
        n_groups = weights.shape[0]
        std_tot = sum(np.square(stds)*weights.reshape(n_groups, 1)) + sum(np.square(dev)*weights.reshape(n_groups, 1))
    else:  
        # one mean value for all channels
        std_tot = sum(np.square(stds)*weights) + sum(np.square(dev)*weights)
        
    return np.sqrt(std_tot/weights_sum)

def min_of_mins(mins):
    """
    Given mins, of N not-overlapping subsets of a whole set, compute the min of the set.
    
    Parameters
    ------------
    mins : Union[list, numpy.ndarray]
        The list of of mins, one for each subset. The single min could be a single value (calculated over the whole subset) or a numpy.ndarray (calculated for each channel).
      
    Returns
    -----------
    Union[numpy.float64, numpy.ndarray]
        The minumum of the whole set (union of subsets).
        The returned minimum could be a single value (calculated over the whole set) or a numpy.ndarray (calculated for each channel), depending on the shape of mins.
    """
    mins = np.array(mins)
    if len(mins[0].shape)==1:
        # mins per channel
        return mins.min(0)
    else:  
        # one mean value for all channels
        return mins.min()

def max_of_maxs(maxs):
    """
    Given maxs, of N not-overlapping subsets of a whole set, compute the max of the set.
    
    Parameters
    ------------
    maxs : Union[list, numpy.ndarray]
        The list of maxs, one for each subset. The single max could be a single value (calculated over the whole subset) or a numpy.ndarray (calculated for each channel).
      
    Returns
    -----------
    Union[numpy.float64, numpy.ndarray]
        The maximum of the whole set (union of subsets). 
        The returned maximum could be a single value (calculated over the whole set) or a numpy.ndarray (calculated for each channel), depending on the shape of maxs.
    """
    maxs = np.array(maxs)
    if len(maxs[0].shape)==1:
        # maxs per channel
        return maxs.max(0)
    else:  
        # one mean value for all channels
        return maxs.max()
    
# usage example

#mean_of_means(stats['mean'], stats['len'])
#std_of_stds(stats['std'], stats['mean'], stats['len'])
#min_of_mins(stats['min'])
#max_of_maxs(stats['max'])

# STATS OVER THE SAMPLE (SONG LEVEL)
def get_song_stats(song, pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=True, return_tensor=False, verbose=False):
    """
    Given a sample entry of EREMUS dataset, returns the statistics of the given sample.
    
    Parameters
    -------------
    song : pandas.core.series.Series.
        It should be a entry of a dataframe, provided by xlsx file *eremus_test.xlsx*.
        **Tips**: open *eremus_test* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    pruned_eeg_root_dir : str
        The eremus raw data directory. 
    return_ch_stats : bool 
        If True it returns channel statistics, if False returns overall statistics.
    return_tensor : bool
        If True it returns statistics as a Tensor, if False returns statistics as numpy.ndarray. 
        **Tips**: use always False (default value).
    verbose : bool
        If true it prints information for debug.
        **Tips**: use always False (default value).
    
    Returns
    ---------------
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        mean of the given song (sample). The first four return values are numpy.ndarray by default; they are numpy.float64 when return_ch_stats is False and return_tensor is False; they are torch.Tensor while return_ternsor is True.
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        standard deviation of the given song (sample).
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        minimum of the given song (sample).
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        maximum of the given song (sample).
    int
        lenght of the given song (sample).   
    """
    eeg_file = song['filename_pruned']
        
    #disable warnings  and get raw eeg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_eeg = mne.io.read_raw_eeglab(pruned_eeg_root_dir + eeg_file, verbose = False)
        start = song['start_index']
        stop = song['end_index']
        raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[start], tmax=raw_eeg.times[stop],  include_tmax=False)
        l = len(raw_eeg)

    # get data 
    raw_eeg = raw_eeg.get_data()
    raw_eeg = interpolate(raw_eeg)
    if return_tensor:
        # convert to Tensor 
        raw_eeg = torch.Tensor(raw_eeg)
    # calculate song-level parameter (OVERALL if dim==None or CHANNEL if dim==1)
    if return_ch_stats:
        mean = raw_eeg.mean(1)
        std = raw_eeg.std(1)
        minim = raw_eeg.min(1)
        maxim = raw_eeg.max(1)
        if return_tensor:
            minim = minim.values
            maxim = maxim.values
    else:
        mean = raw_eeg.mean()
        std = raw_eeg.std()
        minim = raw_eeg.min()
        maxim = raw_eeg.max()
    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, l

# STATS OVER THE SUBJECT (using all songs of the same recording)
def get_subject_stats(eeg_data, sub_id, pruned_eeg_root_dir=pruned_eeg_root_dir, select_single_session=True, select_other_session=False, return_ch_stats=True, verbose=False):
    """
    Given a dataframe containing all entries, subject_id, session type, returns the statistics of the given subject for specified session.
    
    Parameters
    ----------------
    eeg_data : pandas.core.series.Series.
        It should be a entry of a dataframe, provided by xlsx file *eremus_test.xlsx*.
        **Tips**: open *eremus_test* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    sub_id : int
        subject_id to evaluate.
    pruned_eeg_root_dir : str
        The eremus raw data directory. 
    select_single_session : bool
        If True it evaluates OTHER or PERSONAL session, depending on *select_other_session* parameter, otherwise it evaluates both OTHER and PERSONAL sessions.
    select_other_session : bool
        If True it evaluates OTHER session, if False it evaluates PERSONAL session.
    return_ch_stats : bool 
        If True it returns channel statistics, if False returns overall statistics.
    verbose : bool
        If true it prints information for debug.
        **Tips**: use always False (default value).
    
    Returns
    ---------------
    Union[numpy.float64, numpy.ndarray]
        mean of the given subject's songs (all songs). The first four return values are numpy.ndarray by default; they are numpy.float64 when return_ch_stats is False.
    Union[numpy.float64, numpy.ndarray]
        standard deviation of the given subject's songs (all songs).
    Union[numpy.float64, numpy.ndarray]
        minimum of the given subject's songs (all songs).
    Union[numpy.float64, numpy.ndarray]
        maximum of the given subject's songs (all songs).
    int
        lenght of the given subject's songs (all songs).
    """
    # select a particular subject
    sub = eeg_data[eeg_data['subject_id'] == sub_id]
    if select_single_session:
        # select personal or other session
        sub = sub[sub.filename.str.contains('sub'+str(sub_id)+'_ot')==select_other_session]
    # iterate sessions and aggregate measures in list
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'len': []
    }
    for _, song in sub.iterrows():
        m, s, mn, mx, l = get_song_stats(song, pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=return_ch_stats)
        stats['mean'].append(m)
        stats['std'].append(s)
        stats['min'].append(mn)
        stats['max'].append(mx)
        stats['len'].append(l)
    # calculate stats for the whole recording 
    mean, w = mean_of_means(stats['mean'], stats['len'])
    std = std_of_stds(stats['std'], stats['mean'], stats['len'])
    minim = min_of_mins(stats['min'])
    maxim = max_of_maxs(stats['max'])
    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, w

# STATS OVER THE DATASET
def get_dataset_stats(eeg_data, pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=True, verbose=False):
    """
    Given a dataframe containing all entries, returns the statistics of the given dataset.
        
    Parameters
    ----------------
    eeg_data : pandas.core.series.Series.
        It should be a entry of a dataframe, provided by xlsx file *eremus_test.xlsx*.
        **Tips**: open *eremus_test* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    pruned_eeg_root_dir : str
        The eremus raw data directory. 
    return_ch_stats : bool 
        If True it returns channel statistics, if False returns overall statistics.
    verbose : bool
        If true it prints information for debug.
        **Tips**: use always False (default value).
    
    Returns
    ---------------
    Union[numpy.float64, numpy.ndarray]
        mean of the given dataset (all songs - both, personal and other, sessions - of all subjects). The first four return values are numpy.ndarray by default; they are numpy.float64 when *return_ch_stats* is False.
    Union[numpy.float64, numpy.ndarray]
        standard deviation of the given dataset (all songs - both, personal and other, sessions - of all subjects).
    Union[numpy.float64, numpy.ndarray]
        minimum of the given dataset (all songs - both, personal and other, sessions - of all subjects).
    Union[numpy.float64, numpy.ndarray]
        maximum of the given dataset (all songs - both, personal and other, sessions - of all subjects).
    int
        lenght of the given dataset (all songs - both, personal and other, sessions - of all subjects).
    """
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'len': []
    }
    for subject_id in range(0, eeg_data['subject_id'].max()+1):
        # add personal session stats
        m, s, mn, mx, l = get_subject_stats(eeg_data, subject_id, pruned_eeg_root_dir=pruned_eeg_root_dir, select_other_session=False, return_ch_stats=return_ch_stats)
        stats['mean'].append(m)
        stats['std'].append(s)
        stats['min'].append(mn)
        stats['max'].append(mx)
        stats['len'].append(l)
        # add other session stats
        m, s, mn, mx, l = get_subject_stats(eeg_data, subject_id, pruned_eeg_root_dir=pruned_eeg_root_dir, select_other_session=True, return_ch_stats=return_ch_stats)
        stats['mean'].append(m)
        stats['std'].append(s)
        stats['min'].append(mn)
        stats['max'].append(mx)
        stats['len'].append(l)
    # calculate stats for the whole dataset 
    mean, w = mean_of_means(stats['mean'], stats['len'])
    std = std_of_stds(stats['std'], stats['mean'], stats['len'])
    minim = min_of_mins(stats['min'])
    maxim = max_of_maxs(stats['max'])
    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, w

# STATS OVER A BASELINE (BASELINE LEVEL)
def get_subject_stats_by_baseline(baselines, subject_id, required_session_type, pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=True, return_tensor=False, verbose=False):
    """
    Given a subject id and a session type, returns the baseline statistics of the given subject and for the required session type.
    
    Parameters
    -----------
    baselines : pandas.core.series.Series.
        It should be a entry of a dataframe, provided by xlsx file *baselines.xlsx*.
        **Tips**: open *baselines* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    subject_id : int
        The subject_id to evaluate.
    required_session_type : string
        The subject's specific session to evaluate. Only 'other' or 'personal' vaues are allowed.
    pruned_eeg_root_dir : str
        The eremus raw data directory. 
    return_ch_stats : bool 
    return_tensor : bool
        If True it returns statistics as a Tensor, if False returns statistics as numpy.ndarray. 
        **Tips**: use always False (default value).
    verbose : bool
        If true it prints information for debug.
        **Tips**: use always False (default value).
        
    Returns
    ---------------
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        Baseline's mean for the given subject and session type. The first four return values are numpy.ndarray by default; they are numpy.float64 when *return_ch_stats* is False and return_tensor is False; they are torch.Tensor while *return_ternsor* is True.
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        Baseline's standard deviation for the given subject and session type.
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        Baseline's minimum for the given subject and session type.
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        Baseline's maximum for the given subject and session type.
    int
        Baseline's lenght for the given subject and session type.   
    """
    # select and extract baselines for subject_id
    subject_baselines = baselines[baselines['sub_id'] == subject_id]
    # select infos (use_other and use_eyes_closed) from required_session
    if not (required_session_type=='other' or required_session_type=='personal'):
        raise ValueError("Please provide a valid required_session_type: 'other' or 'personal' are allowed.")
    required_baseline = subject_baselines[subject_baselines['rec_type'] == required_session_type]
    use_other = required_baseline['use_other'].bool()
    use_eyes_closed = required_baseline['use_eyes_closed'].bool()
    
    # decide file to open, depending on extracted infos
    if use_other:
        baseline = subject_baselines[subject_baselines['rec_type'] == 'other']
    else:
        baseline = subject_baselines[subject_baselines['rec_type'] == 'personal']
        
    eeg_file = baseline['pruned_filename'].item()
    
    if verbose:
        print('\n[SUBJECT SELECTION]')
        print(subject_baselines)
        print('\n[SESSION SELECTION]')
        print(required_baseline)
        print('\n[INFOS FROM SESSION]')
        print(use_other, use_eyes_closed)
        print('\n[FILE SELECTION BASED ON INFOS]')
        print(eeg_file)

    #disable warnings  and get raw eeg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_eeg = mne.io.read_raw_eeglab(pruned_eeg_root_dir + eeg_file, verbose = False)
        if use_eyes_closed:
            start = baseline['cs'].item()
            stop = baseline['ce'].item()
        else:
            start = baseline['os'].item()
            stop = baseline['oe'].item()
        raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[start], tmax=raw_eeg.times[stop],  include_tmax=False)
        l = len(raw_eeg)

    # get data 
    raw_eeg = raw_eeg.get_data()
    raw_eeg = interpolate(raw_eeg)
    if return_tensor:
        # convert to Tensor 
        raw_eeg = torch.Tensor(raw_eeg)
    # calculate baseline-level parameter (OVERALL if dim==None or CHANNEL if dim==1)
    if return_ch_stats:
        mean = raw_eeg.mean(1)
        std = raw_eeg.std(1)
        minim = raw_eeg.min(1)
        maxim = raw_eeg.max(1)
        if return_tensor:
            minim = minim.values
            maxim = maxim.values
    else:
        mean = raw_eeg.mean()
        std = raw_eeg.std()
        minim = raw_eeg.min()
        maxim = raw_eeg.max()
    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, l


# STATS OVER THE DATASET (using all baselines)
def get_dataset_stats_by_baseline(baselines, pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=True, verbose=False):
    """
    Given a dataframe containing all entries (baselines), returns the statistics of the given dataset.
        
    Parameters
    ----------------
    baselines : pandas.core.series.Series.
        It should be a entry of a dataframe, provided by xlsx file *baselines.xlsx*.
        **Tips**: open *baselines* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    pruned_eeg_root_dir : str
        The eremus raw data directory. 
    return_ch_stats : bool 
        If True it returns channel statistics, if False returns overall statistics.
    verbose : bool
        If true it prints information for debug.
        **Tips**: use always False (default value).
    
    Returns
    ---------------
    Union[numpy.float64, numpy.ndarray]
        mean of the given dataset (all baselines - of both, personal and other, sessions - of all subjects) . The first four return values are numpy.ndarray by default; they are numpy.float64 when *return_ch_stats* is False.
    Union[numpy.float64, numpy.ndarray]
        standard deviation of the given dataset (all baselines - of both, personal and other, sessions - of all subjects) .
    Union[numpy.float64, numpy.ndarray]
        minimum of the given dataset (all baselines - of both, personal and other, sessions - of all subjects) .
    Union[numpy.float64, numpy.ndarray]
        maximum of the given dataset (all baselines - of both, personal and other, sessions - of all subjects) .
    int
        lenght of the given dataset (all baselines - of both, personal and other, sessions - of all subjects) .
    """
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'len': []
    }
    for subject_id in range(0, baselines['sub_id'].max()+1):
        # add personal session stats
        m, s, mn, mx, l = get_subject_stats_by_baseline(baselines, subject_id, 'personal', pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=return_ch_stats)
        stats['mean'].append(m)
        stats['std'].append(s)
        stats['min'].append(mn)
        stats['max'].append(mx)
        stats['len'].append(l)
        # add other session stats
        m, s, mn, mx, l = get_subject_stats_by_baseline(baselines, subject_id, 'other', pruned_eeg_root_dir=pruned_eeg_root_dir, return_ch_stats=return_ch_stats)
        stats['mean'].append(m)
        stats['std'].append(s)
        stats['min'].append(mn)
        stats['max'].append(mx)
        stats['len'].append(l)
    # calculate stats for the whole dataset 
    mean, w = mean_of_means(stats['mean'], stats['len'])
    std = std_of_stds(stats['std'], stats['mean'], stats['len'])
    minim = min_of_mins(stats['min'])
    maxim = max_of_maxs(stats['max'])
    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, w

#======================================================
# STATS USED WITH EXTRACTED FEATURES (SUBJECT LEVEL)
#======================================================

def get_fe_stats(sample, root_dir=preprocessed_eeg_root_dir/Path('de'), axis=None, verbose=False):
    """
    Given a sample entry of EREMUS dataset, returns the statistics of the given sample.

    Parameters
    ----------------
    sample : pandas.core.series.Series
        It should be a entry of a dataframe, provided by xlsx file *eremus_test.xlsx*.
        **Tips**: open *eremus_test* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    root_dir : str
        The eremus raw data directory (.npz format). 
    axis : Union[int, tuple of int]
        Axis used in statistics computations. Assuming FE data are of shape *(C, F, B)*, use **None** (default value) in order to return a single value; use **1** in order to return stats over F axis (i.e. stats are per channel, per band); use **(0, 1)** in order to return stats over (C, F) axis (i.e. stats are per band). Assuming FE data are of shape *(C, F)*, use **None** (default value) in order to return a single value; use **1** in order to return stats over F axis (i.e. stats are per channel).
    return_tensor : bool
        If True it returns statistics as a Tensor, if False returns statistics as numpy.ndarray. 
        **Tips**: use always False (default value).
    verbose : bool
        If true it prints information for debug.
        **Tips**: use always False (default value).

    Returns
    -----------
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        mean of the given song (sample). The first four return values are numpy.ndarray by default; they are numpy.float64 when return_ch_stats is False and return_tensor is False; they are torch.Tensor while return_ternsor is True.
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        standard deviation of the given song (sample).
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        minimum of the given song (sample).
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        maximum of the given song (sample).
    int
        lenght of the given song (sample).   
    """
    filename = root_dir + str(sample.name) + '.npz'
        
    # get numpy vector
    data = np.load(filename)['arr_0']
    C, F, B = data.shape #(n_channels, n_features, n_bands)
        
    mean = data.mean(axis = axis) 
    std = data.std(axis = axis) 
    minim = data.min(axis = axis) 
    maxim = data.max(axis = axis)

    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, F

def get_fe_subject_stats(eeg_data, sub_id, root_dir=preprocessed_eeg_root_dir/Path('de'), select_single_session=True, select_other_session=False, axis=None, verbose=False):
    """
    Given a dataframe containing all entries, subject_id, session type, returns the statistics of the given subject for the specified session.
    
    Parameters
    ------------
    eeg_data : pandas.core.series.Series.
        It should be a entry of a dataframe, provided by xlsx file *eremus_test.xlsx*.
        **Tips**: open *eremus_test* file with pandas and access the returned dataframe with iloc in order to return pandas.core.series.Series.
    sub_id : int
        subject_id to evaluate.
    pruned_eeg_root_dir : str
        The raw data directory (.npz format)
    select_single_session : bool
        If True it evaluates OTHER or PERSONAL session, depending on *select_other_sessioné* parameter, otherwise it evaluates both OTHER and PERSONAL sessions.
    select_other_session : bool
        If True it evaluates OTHER session, if False it evaluates PERSONAL session.
    axis : Union[int, tuple of int]
        Axis used in statistics computations. Assuming FE data are of shape *(C, F, B)*, use **None** (default value) in order to return a single value; use **1** in order to return stats over F axis (i.e. stats are per channel, per band); use **(0, 1)** in order to return stats over (C, F) axis (i.e. stats are per band). Assuming FE data are of shape *(C, F)*, use **None** (default value) in order to return a single value; use **1** in order to return stats over F axis (i.e. stats are per channel).
    verbose : bool
        If true it prints information for debug. **Tips**: use always False (default value).
    
    Returns
    -----------
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        mean of the given subject (all songs). The first four return values are numpy.ndarray by default; they are numpy.float64 when return_ch_stats is False and return_tensor is False; they are torch.Tensor while return_ternsor is True.
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        standard deviation of the given subject (all songs).
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        minimum of the given subject (all songs).
    Union[numpy.float64, numpy.ndarray, torch.Tensor]
        maximum of the given subject (all songs).
    int
        lenght of the given subject (all songs).   
        
    Examples
    --------------
    >>> import json
    >>> from pathlib import Path
    >>> from eremus.preprocessing import preprocessing as pp
    >>> # configure eremus Path
    >>> configuration_path = Path(__file__).parent.parent
    >>> with open(configuration_path / 'configuration.txt') as json_file:
    ...     configuration = json.load(json_file)
    ...     path_to_eremus_data = configuration['path_to_eremus_data']
    >>> dataset_file = pd.read_excel(path_to_eremus_data + 'eremus_test.xlsx')
    >>> pruned_eeg_root_dir = path_to_eremus_data + 'recordings_pruned_with_ICA\\'
    >>> preprocessed_eeg_root_dir = path_to_eremus_data + 'preprocessed_data\\'
    >>> 
    >>>     stats_all_bands = {
    ...     'mean': [],
    ...     'std': [],
    ...     'min': [],
    ...     'max': [],
    ...     'weights': []
    ... }
    >>> 
    >>> stats_per_band = {
    ...     'mean': [],
    ...     'std': [],
    ...     'min': [],
    ...     'max': [],
    ...     'weights': []
    ... }
    >>> 
    >>> for sub in range(34):
    ... 
    ...     stats = pp.get_fe_subject_stats(samples, sub, root_dir=preprocessed_eeg_root_dir + "de\\", select_single_session=False)
    ...     for k, v in zip(stats_all_bands.keys(), stats):
    ...         stats_all_bands[k] = stats_all_bands[k] + [v]
    ... 
    ...     stats = pp.get_fe_subject_stats(samples, sub, root_dir=preprocessed_eeg_root_dir + "de\\", select_single_session=False, axis=(0, 1))
    ...     for k, v in zip(stats_per_band.keys(), stats):
    ...         stats_per_band[k] = stats_per_band[k] + [v]

    """
    if axis==1:
        warnings.warn("Please keep attention. If your data are of shape (C, F, B), don't use 1 axis: Not Implemented Yet.")
    # select a particular subject
    sub = eeg_data[eeg_data['subject_id'] == sub_id]
    if select_single_session:
        # select personal or other session
        sub = sub[sub.filename.str.contains('sub'+str(sub_id)+'_ot')==select_other_session]
    # iterate sessions and aggregate measures in list
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'len': []
    }
    for _, song in sub.iterrows():
        m, s, mn, mx, l = get_fe_stats(song, root_dir=root_dir, axis=axis)
        stats['mean'].append(m)
        stats['std'].append(s)
        stats['min'].append(mn)
        stats['max'].append(mx)
        stats['len'].append(l)
    # calculate stats for the whole recording
    mean, w = mean_of_means(stats['mean'], stats['len'])
    std = std_of_stds(stats['std'], stats['mean'], stats['len'])
    minim = min_of_mins(stats['min'])
    maxim = max_of_maxs(stats['max'])
    if verbose:
        print('mean: ' + str(mean), end='\t')
        print('std: ' + str(std), end='\t')
        print('minim: ' + str(minim), end='\t')
        print('maxim: ' + str(maxim))
    return mean, std, minim, maxim, w

# Open data using eremus_test file
# xls_file = 'eremus_test.xlsx'
# eeg_data = pd.read_excel(xls_file)
# get_fe_subject_stats(eeg_data, 14, root_dir='preprocessed\\de\\', select_single_session=False) #axis=(0, 1))

if __name__ == "__main__":
    
    from EremusDataset import EremusDataset_V2, RandomCrop_V2, SetMontage_V2, ToArray_V2, ToMatrix_V2, ToTensor_V2

    # define transforms
    #crop = RandomCrop_V2(1280)
    set_montage = SetMontage_V2()
    to_array = ToArray_V2()
    to_tensor = ToTensor_V2(interface='unpacked_values', label_interface='long')

    # compose transforms
    composed = transforms.Compose([set_montage, to_array])

    # load dataset
    emus = EremusDataset_V2(xls_file='eremus_test.xlsx',
                                  eeg_root_dir='eeglab_raws\\',
                                  data_type=EremusDataset_V2.DATA_PRUNED,
                                  transform=composed)

    # normalization covers standardization effects and viceversa
    sample = emus[0]['eeg']

    ns = channnel_z_score_norm(channel_minmax_norm(sample, -1, 1))
    sn = channel_minmax_norm(channnel_z_score_norm(sample), -1, 1)

    print(ns.std(), ns.mean(), ns.max(), ns.min())
    print(sn.std(), sn.mean(), sn.max(), sn.min())

    # Test functions
    
    # Open data using eremus_test file
    xls_file = 'eremus_test.xlsx'
    eeg_data = pd.read_excel(xls_file)
    # Open data using baselines file
    xls_file = 'baselines.xlsx'
    baselines = pd.read_excel(xls_file)

    #STATS OVER THE SAMPLE (SONG LEVEL)
    get_song_stats(eeg_data.iloc[0], return_ch_stats=True)

    # STATS OVER THE SUBJECT (using all songs of the same recording)
    get_subject_stats(eeg_data, 14, select_other_session=False, return_ch_stats=False)
    get_subject_stats(eeg_data, 14, select_single_session=False, return_ch_stats=False)

    # STATS OVER THE DATASET
    get_dataset_stats(eeg_data, return_ch_stats=False)
    
    # SUBJECTS STATS BASED ON BASELINE
    get_subject_stats_by_baseline(baselines, 14, 'personal')
    
    # DATASET STATS BASED ON BASELINE
    get_dataset_stats_by_baseline(baselines, return_ch_stats=False)
    
    #SUBJECT STATS ON FEATURES
    get_fe_subject_stats(eeg_data, 14, root_dir='preprocessed\\de\\',select_single_session=False) #axis=(0, 1))


