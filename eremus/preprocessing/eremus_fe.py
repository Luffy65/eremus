"""
This module implements some useful functions, in order to extract features from EEG data.
It is intended to complete the *mne_features* library, looking at different features in accordance to Hong He et al. (2020).

You could use this module, to extract the implemented features, without using *mne_features*, but you can also use the latter.

Examples
--------------
>>> # Single feature extraction
>>> import mne
>>> import numpy as np
>>> import pandas as pd
>>> from random import randint
>>> from eremus.preprocessing import eremus_fe
>>> 
>>> dataset_directory = '..\\EREMUSDataset\\'
>>> dataset = pd.read_excel(dataset_directory + 'eremus_test.xlsx')
>>> filename = dataset.iloc[randint(0, len(dataset) - 1)].filename_pruned
>>> data_directory = '..\\EREMUSDataset\\recordings_pruned_with_ICA\\'
>>> raw = mne.io.read_raw_eeglab(data_directory + filename, verbose=False)
>>> np_data = raw.get_data()
>>> np_data.shape
(32, 10249)
>>> eremus_fe.compute_channel_covariance(np_data)

>>> # Use mne_features
>>> from mne_features.feature_extraction import extract_features
>>> # Example with continous data
>>> 
>>> # Get data from raw
>>> data = raw.get_data()
>>> # Get shape of data
>>> C, T = data.shape
>>> # Add Epoch dimension
>>> data = data.reshape(1, C, T)
>>> 
>>> # Get Sampling Frequency
>>> sfreq = raw.info['sfreq']
>>> 
>>> # Select custom functions
>>> selected_funcs = [('covariance', eremus_fe.compute_covariance), 
>>>                   ('mean_absolute_deviation', eremus_fe.compute_mean_absolute_deviation), 
>>>                   ('median', eremus_fe.compute_median), 
>>>                   ('median_absolute_deviation', eremus_fe.compute_median_absolute_deviation), 
>>>                   ('maximum', eremus_fe.compute_maximum), 
>>>                   ('minimum', eremus_fe.compute_minimum), 
>>>                   ('upper_quartile', eremus_fe.compute_upper_quartile), 
>>>                   ('lower_quartile', eremus_fe.compute_lower_quartile)]
>>> 
>>> # Features Maps
>>> F = len(selected_funcs)
>>> 
>>> # Extract features (chan axis concatenation - for continous data)
>>> features = extract_features(data, sfreq, selected_funcs, return_as_df=False).reshape(F, C)
>>> features.shape
(8, 32)

See also
--------------
mne_features : https://mne.tools/mne-features/generated/mne_features.feature_extraction.extract_features.html

References
---------------
Hong He, Yonghong Tan, Jun Ying, Wuxiong Zhang,
Strengthen EEG-based emotion recognition using firefly integrated optimization algorithm,
Applied Soft Computing,
Volume 94,
2020,
106426,
ISSN 1568-4946,
https://doi.org/10.1016/j.asoc.2020.106426.
"""
from scipy.stats import median_abs_deviation

import mne
import numpy as np
import pandas as pd
import mne_features

from sklearn import svm
from eremus.eremus_utils import Session
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mne_features.feature_extraction import extract_features, FeatureExtractor

# Symmetric Electrodes
symmetric_electrodes_names = {
 'Fp1': 'Fp2',
 'F7': 'F8',
 'F3': 'F4',
 'FC1': 'FC2',
 'C3': 'C4',
 'FC5': 'FC6',
 'FT9': 'FT10',
 'T7': 'T8',
 'CP5': 'CP6',
 'CP1': 'CP2',
 'P3': 'P4',
 'P7': 'P8',
 'PO9': 'PO10',
 'O1': 'O2'}

symmetric_electrodes = {
    Session.chs[k]: Session.chs[v]
    for k, v in symmetric_electrodes_names.items()
}

def compute_channel_covariance(data):
    """Covariance of the data (per channel, beetween channels).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels, n_channels)
        The covariance beetween channels.
    
    Notes
    -----
    Alias of the feature function: **channel_covariance**
    """
    return np.cov(data, ddof=1)

def compute_covariance(data, k=128):
    """Covariance of the data (per channel, beetween samples at distance *k*).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    k : int
        distance beetween samples. Default to 128.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The covariance beetween samples at distance *k*.
    
    
    Notes
    -----
    Alias of the feature function: **covariance**
    """
    # get shape
    n_channels, n_times = data.shape
    assert n_times > k, 'Please provide a k value smaller than ' + str(n_times)
    
    # init covariance
    cov = np.zeros(n_channels)
    
    # iter channels
    for ch in range(n_channels):
        # select current time-points
        t = data[ch,  0:n_times-k]
        # select future time-points (current+k)
        t_k = data[ch, k:n_times]
        # calculate covariance
        cov[ch] = np.cov(t, t_k, ddof=1)[0, 1]

    return cov

def compute_mean_absolute_deviation(data):
    """Mean absolute deviation of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The mean absolute deviation (per channel).
    
    Notes
    -----
    Alias of the feature function: **mean_absolute_deviation**
    """
    return np.sum(data - np.mean(data, axis=-1).reshape(-1, 1), axis=-1)/data.shape[-1]

def compute_median(data):
    """Median of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The median (per channel).
    
    Notes
    -----
    Alias of the feature function: **median**
    """
    return np.median(data, axis=-1)

def compute_median_absolute_deviation(data):
    """Median absolute deviation of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The median absolute deviation (per channel).
    
    Notes
    -----
    Alias of the feature function: **median_absolute_deviation**
    """
    return median_abs_deviation(data, axis=-1)

def compute_maximum(data):
    """Maximum of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The maximum (per channel).
    
    Notes
    -----
    Alias of the feature function: **maximum**
    """
    return np.max(data, axis=-1)

def compute_minimum(data):
    """Minimum of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The minimum (per channel).
    
    Notes
    -----
    Alias of the feature function: **minimum**
    """
    return np.min(data, axis=-1)

def compute_upper_quartile(data):
    """Upper Quartile of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The upper quartile (per channel).
    
    Notes
    -----
    Alias of the feature function: **upper_quartile**
    """
    return np.quantile(data, 0.75, axis=-1)

def compute_lower_quartile(data):
    """Lower Quartile of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    
    Returns
    -------
    numpy.ndarray, shape (n_channels,)
        The lower quartile (per channel).
    
    Notes
    -----
    Alias of the feature function: **lower_quartile**
    """
    return np.quantile(data, 0.25, axis=-1)


# ### Frequency Domain Features
# 
# Based on Hong He et al. (2020):
# The following features should be calculated over 5 different frequency bands, delta (δ: 1–4 Hz), theta (θ: 4–8 Hz), alpha(α: 8–13 Hz), beta (β:13–30 Hz), and gamma (γ : 31–50 Hz):
# - Power Spectral Density 
# - Band Power
# - Band Power Ratio
# 
# All these features could be computated using mne_features functios:
# - Power Spectral Density --> just wrap utils.power_spectrum function, bands are not equal in length, so we can compute PSD over all the spectrum
# - Band Power --> compute_pow_freq_bands with normalize=False,
# - Band Power Ratio --> compute_pow_freq_bands with normalize=True (default use)

def compute_psd(data, sfreq=128, flatten=False):
    """PSD of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    sfreq : int
        Sampling Frequency in Hz. Default to 128.
    flatten : bool
        if True, flatten the output array
    
    Returns
    -------
    numpy.ndarray 
        The power spectral density of input data. Its shape is *(n_channels, n_freqs)* if flatten is False (default), then *(n_channels\*n_freqs,)* if flatten is True.
    
    Notes
    -----
    Alias of the feature function: **psd**
    """
    psd, _ = mne_features.utils.power_spectrum(sfreq, data, fmin=0.5, fmax=sfreq/2)
    if flatten:
        psd = psd.reshape(-1)
    return psd

def compute_pow_bands(data, sfreq=128, freq_bands=np.array([0.5, 4.0, 8.0, 13.0, 30.0, 64.0]), flatten=False):
    """Power Bands of the data (per channel).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    sfreq : int
        Sampling Frequency in Hz. Default to 128.
    freq_bands : numpy.ndarray or dict (default: np.array([.5, 4, 8, 13, 30, 64]))
        Array or dict containing the frequency bands edges. We suggest to not modify it. Otherwise see *mne_features* documentation for further details.
    flatten : bool
        if True, flatten the output array
    
    Returns
    -------
    numpy.ndarray 
        The power bands of input data. Its shape is *(n_channels, n_bands)* if flatten is False (default), then *(n_channels\*n_bands,)* if flatten is True. *n_bands* is equal to the shape of *freq_bands*, minus 1.
    
    Notes
    -----
    Alias of the feature function: **pow_bands**
    
    See also
    ----------
    compute_pow_freq_bands: https://mne.tools/mne-features/generated/mne_features.univariate.compute_pow_freq_bands.html
    """
    C, _ = data.shape
    if flatten:
        return mne_features.univariate.compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands, normalize=False)
    else:
        return mne_features.univariate.compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands, normalize=False).reshape(C, -1)

def compute_differential_pow_bands(data, sfreq=128, freq_bands=np.array([0.5, 4.0, 8.0, 13.0, 30.0, 64.0]), flatten=False, apply_padding=True):
    """Differential Bands Power of the data (per symmetric channels).
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        A array representing EEG data, being *n_channels* the number of EEG channels and *n_times* the number of time-points.
    sfreq : int
        Sampling Frequency in Hz. Default to 128.
    freq_bands : numpy.ndarray or dict (default: np.array([.5, 4, 8, 13, 30, 64]))
        Array or dict containing the frequency bands edges. We suggest to not modify it. Otherwise see *mne_features* documentation for further details.
    flatten : bool
        if True, flatten the output array.
    apply_padding : bool
        if True, padding is added to the output array
    
    Returns
    -------
    numpy.ndarray 
        The differential band power. Power bands of symmetric channels are subctracted each other.
        The output shape depends on *flatten* and *apply_padding* parameters:
        
            - shape (n_channels/2, n_bands) if !flatten and apply_padding (default)
            - shape (n_channels/2*n_bands,) if flatten and apply_padding
            - shape (n_sym, n_bands) if !flatten and !apply_padding 
            - shape (n_sym*n_bands,) if flatten and !apply_padding
    
    Notes
    -----
    Alias of the feature function: **differential_pow_bands**
    
    See also
    ----------
    compute_pow_freq_bands: https://mne.tools/mne-features/generated/mne_features.univariate.compute_pow_freq_bands.html
    """
    C, _ = data.shape
    B = freq_bands.shape[0] - 1
    bp = mne_features.univariate.compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands, normalize=False).reshape(C, B)
    
    dbp = np.zeros((C//2, B))
    for channel, symmetric in symmetric_electrodes.items():
        dbp[channel - 1] = bp[channel, :] - bp[symmetric, :]
        
    if not apply_padding:
        dbp = dbp[1:-1, :]
    
    return dbp.reshape(-1) if flatten else dbp 

# already implemented temporal features
temporal_features = {
     'hjorth_complexity',
     'hjorth_mobility',
     'kurtosis',
     'mean',
     'ptp_amp',
     'skewness',
     'std',
     'variance',
     'zero_crossings'}

# Other frequency domain features
frequency_domain_features = {
 'app_entropy',
 'decorr_time',
 'energy_freq_bands',
 'higuchi_fd',
 'hjorth_complexity_spect',
 'hjorth_mobility_spect',
 'hurst_exp',
 'katz_fd',
 'line_length',
 'samp_entropy',
 'spect_edge_freq',
 'spect_entropy',
 'spect_slope',
 'svd_entropy',
 'svd_fisher_info',
 'teager_kaiser_energy',
 'wavelet_coef_energy'}

def compute_differential_entropy(raw, l, h, window_size=32):
    """
    Gets and prints the spreadsheet's header columns
    
    Parameters
    ----------
    raw: mne.io.RawArray
        The input raw eeg
    l: int
        Low frequency bound, used to filter raw
    h: int
        High frequency bound, used to filter raw
    window_size: int
        Number of timepoints in a window during DE computation

    Returns
    -------
    numpy.ndarray
        a feature array of shape (C, F)
        
    Examples
    -------------
    **Compute Differential entropy for a single sample**
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from eremus.eremus_utils import get_raw
    >>> from eremus.preprocessing import eremus_fe
    >>> 
    >>> # Load sample and convert to mne
    >>> sample = dataset.iloc[0]
    >>>
    >>> # Get filename
    >>> raw_fname = data_directory + sample.filename_pruned # Change filename to get different data types
    >>> raw = get_raw(raw_fname, sample.start_index, sample.end_index)
    >>>
    >>> first=True
    >>> # Filter each band and compute de
    >>> for (l,h) in [(1,3),(4,7),(8,13),(14,30),(31,50)]:
    >>>
    >>>     de_band = eremus_fe.compute_differential_entropy(raw, l, h)
    >>>
    >>>     if first==True:
    >>>         de=de_band
    >>>         first=False
    >>>     else:
    >>>         de=np.concatenate((de,de_band),axis=2)
    >>> 
    >>>     # Here differential entropy for a single band is of shape (C, F, 1)
    >>> # Here differential entropy is of shape (C, F, B)
    >>> # Save recording
    >>> recording_out_path = 'your-path-and-filename.npz'
    >>> np.savez_compressed(recording_out_path, de)
    
    **Compute differential entropy with sliding window for all dataset**
    
    >>> import os
    >>> import numpy as np
    >>> import pandas as pd
    >>> import mne; mne.set_log_level(False)
    >>> 
    >>> # Set Params
    >>>
    >>> # Sampling frequency
    >>> sfreq = 128
    >>>
    >>> # Windowing Parameters
    >>> window_size = 5.0
    >>> step_size = 1.0
    >>> overlap_size = window_size - step_size
    >>> epsilon = 1/sfreq
    >>>    
    >>> # Your Path ro EREMUS Dataset
    >>> path_to_eremus = 'your-path\\'
    >>>
    >>> # Create Output Directory
    >>> # Set write dir
    >>> output_dir = path_to_eremus + "preprocessed\\de_ws\\" + str(int(window_size)) + '_' + str(int(step_size)) + '\\'
    >>> # Create output dir
    >>> os.makedirs(output_dir, exist_ok=True)
    >>>
    >>> # Process each recording
    >>> for i, sample in pd.read_excel(path_to_eremus + 'eremus_test.xlsx').iterrows():
    >>>
    >>>     # Load sample and convert to mne
    >>>
    >>>     # Get filename
    >>>     raw_fname = root_dir + sample.filename_pruned
    >>>     # Get raw object
    >>>     raw = get_raw(raw_fname, sample.start_index, sample.end_index)
    >>>
    >>>     # Create Epochs
    >>>
    >>>     # Extract fake events (Window Sliding)
    >>>     events = mne.make_fixed_length_events(raw, duration=window_size, overlap=overlap_size)
    >>>     # Extract epochs and get data
    >>>     epochs = mne.Epochs(raw, events, 1, 0, window_size - epsilon, proj=True,
    ...                         baseline=None, preload=True, verbose=False)
    >>>
    >>>     de_dict = {}
    >>>     for j, epoch in enumerate(epochs):
    >>>
    >>>         # wrap epoch data in a raw object
    >>>         raw = mne.io.RawArray(epoch, epochs.info, verbose = False)
    >>>
    >>>         first=True
    >>>         # Filter each band and compute de
    >>>         for (l,h) in [(1,3),(4,7),(8,13),(14,30),(31,50)]:
    >>>
    >>>             de_band = compute_differential_entropy(raw, l, h)
    >>>
    >>>             if first==True:
    >>>                 de=de_band
    >>>                 first=False
    >>>             else:
    >>>                 de=np.concatenate((de,de_band),axis=2)
    >>>
    >>>             # Here differential entropy for a single band is of shape (C, F, 1)
    >>>
    >>>         # Here differential entropy is of shape (C, F, B)
    >>>         de_dict['arr_'+str(j)] = de
    >>>
    >>>     # Save recording
    >>>     recording_out_path = output_dir + str(i) + '.npz'
    >>>     args = (v for _, v in de_dict.items())
    >>>     np.savez_compressed(recording_out_path, *args)
    """
    
    #filter band of interest
    raw.filter(l, None, method='iir', iir_params={"order":2, "ftype":"butter"})
    raw.filter(None, h, method='iir', iir_params={"order":2, "ftype":"butter"})
    
    # Convert MNE to numpy array
    band = raw.get_data()
    _, n_timepoints = band.shape 
    
    #Compute DE for windows of window_size time points without overlap
    start = range(0, n_timepoints, window_size) 
    
    for start in range(0, n_timepoints, window_size)[:-1]:
        # Compute variance
        var = np.var(band[:, start:start+window_size],axis=1)
        # DE partial computation
        de_tmp=np.array([0.5*np.log(2*np.pi*np.e*v) if v!=0 else 0 for v in var])
        de_tmp=np.expand_dims(de_tmp,axis=1)
        
        # Check first iteration
        if start==0:
            de_band=de_tmp
        else:
            de_band=np.concatenate((de_band,de_tmp),axis=1)
            
    return np.expand_dims(de_band,axis=2)