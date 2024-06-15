#!/usr/bin/env python
# coding: utf-8

import os
import re
import mne
import pandas as pd
import matplotlib.pyplot as plt

class Session(object):
    """
    A class that describes a listening session, during EREMUS data acquisition.
    
    Arguments
    ---------------
    subject_id : int
        Subject identifier for that session.
    edf_file : str
        Path to the eeg data. Only edf format is supported
    s_type : str
        Session type, could be *personal* or *other*
    """
    
    """
    A dictionary containing channel names as keys and channel indices as values.
    """
    chs = {'Cz': 0,
         'Fz': 1,
         'Fp1': 2,
         'F7': 3,
         'F3': 4,
         'FC1': 5,
         'C3': 6,
         'FC5': 7,
         'FT9': 8,
         'T7': 9,
         'CP5': 10,
         'CP1': 11,
         'P3': 12,
         'P7': 13,
         'PO9': 14,
         'O1': 15,
         'Pz': 16,
         'Oz': 17,
         'O2': 18,
         'PO10': 19,
         'P8': 20,
         'P4': 21,
         'CP2': 22,
         'CP6': 23,
         'T8': 24,
         'FT10': 25,
         'FC6': 26,
         'C4': 27,
         'FC2': 28,
         'F4': 29,
         'F8': 30,
         'Fp2': 31
        }

    def __init__(self, subject_id, edf_file, s_type):
        if subject_id<0 or subject_id>33:
            raise ValueError('Please provide a valid subject ID. Only integers in [0, 33] range are allowed.')
        if not edf_file.endswith('.edf'):
            raise ValueError('Only edf files coul be opened in Session class')
        if s_type not in ['personal', 'other']:
            raise ValueError('Please provide a valid session type. Only "personal" and "other" values are allowed.')
        self.subject_id = subject_id
        self.eeg = mne.io.read_raw_edf(edf_file)
        self.s_type = s_type
        
    def pick_data(self):
        """
        Picks only data channels.
        """
        #pick data channels
        chs = self.eeg.ch_names[4:36]
        self.eeg.pick_channels(chs)
    
    def set_montage(self):
        """
        Sets the correct montage to the session. 10-20 international system is used for locations. In EREMUS Dataset, subjects with ID in range [0, 5], have some data channels inverted. This function makes all appropriate corrections.
        """
        if(self.subject_id<6):
            data = self.eeg.get_data()
            data[[self.chs['Fz'], self.chs['FC2'], self.chs['C4']], :] = data[[self.chs['C4'], self.chs['Fz'], self.chs['FC2']], :]
            info = self.eeg.info
            new_eeg = mne.io.RawArray(data, info, verbose=False)
            self.eeg = new_eeg
        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        self.eeg.set_montage(ten_twenty_montage)

    def hp_filter(self):
        """
        It applies a High-pass Filter to the Raw data. The lower edge is 0.5 Hz.
        """
        #load data
        self.eeg.load_data()
        # delete continous components
        fig = self.eeg.filter(0.5, None)
        
    def plot(self):
        """
        Plot data (time-series).
        """
        self.eeg.plot()

    def get_eeg(self):
        """
        Get the eeg file
        
        Returns
        -----------
        mne.io.RawEDF
            The Raw edf file
        """
        return self.eeg

# define functions for oder lists
def __atoi__(text):
    return int(text) if text.isdigit() else text

def __natural_keys__(text):
    return [ __atoi__(c) for c in re.split(r'(\d+)', text) ]

def __invertPersonalAndOther__(sessions):
    for sub_id in range(int(len(sessions)/2)):
        ot = sub_id*2
        pr = ot + 1
        temp = sessions[ot]
        sessions[ot] = sessions[pr]
        sessions[pr] = temp
    return sessions

def getRecordings(root_dir, file_ext='edf'):
    """
    It scans *root_dir* in order to find file that have the *file_ext* extension.
    Found files are ordered by natural keys.
    
    Parameters
    ------------
    root_dir : str
        The directory to scan.
    file_ext : str
        file extension to find in *root_dir*. Extension must be a strind without the dot (e.g 'set' and not '.set').
        
    Returns
    ----------
    list of str
        An ordered list of file names in the given directory. The returned list could be easy accessed with *sub* and *sub_ot* functions.
        
    See also
    -----------
    getSessions : alias for getRecordings
    getPrunedSessions: a particular implementation for pruned file directory
    sub : function to access the returned list (personal sessions)
    sub_ot : function to access the returned list (other sessions)
    """
    # define root directory for recordings
    root = root_dir
    # list all recordings
    recordings = os.listdir(root)
    print("In " + root + " there are " + str(len(recordings)) + " files")
    # compile regex for search file_ext files
    ext = re.compile(r'.'+file_ext)
    # select only edf files
    recordings = list(filter(lambda file: (ext.search(file)!=None), recordings)) 
    # natural sort of recordings
    recordings.sort(key=__natural_keys__)
    print("Getting " + file_ext +" files...\n" + str(len(recordings)) + " found.")
    return recordings

def getSessions(root_dir, file_ext='edf'):
    """
    Alias of *getRecordings*.
        
    See also
    -----------
    getRecordings : alias for getSessions

    """
    return getRecordings(root_dir, file_ext=file_ext)

def getPrunedSessions(root_dir):
    """
    It scans *root_dir* in order to find file that have the *.set* extension.
    Found files are ordered in such a way to be accessed with *sub* and *sub_ot* functions.
    
    Parameters
    ------------
    root_dir : str
        The directory to scan.
        
    Returns
    ----------
    list of str
        An ordered list of file names in the given directory. The returned list could be easy accessed with *sub* and *sub_ot* functions.
        
    See also
    -----------
    getSessions: a particular implementation for not pruned file directory
    sub : function to access the returned list (personal sessions)
    sub_ot : function to access the returned list (other sessions)
    """
    return __invertPersonalAndOther__(getRecordings(root_dir, file_ext='set'))

def sub(sub_id, other = False):
    """
    Get the index for personal session of subject *sub_id*.
    It assumes that index will access a list returned from *getSessions* or *getPrunedSessions*.
    
    Parameters
    --------------
    sub_id : int
        A subject id in [0, 33] range
        
    Returns
    ----------
    int
        The index for access a list returned from *getSessions* or *getPrunedSessions*.
        
    Examples
    -------------
    >>> sessions = getSessions('path-to-eremus-data-dir')
    >>> # get the personal session of subject 15
    >>> sessions[sub(15)]
    """
    if other:
        return sub_id*2 + 1
    else:
        return sub_id*2
    
def sub_ot(sub_id):
    """
    Get the index for other session of subject *sub_id*.
    It assumes that index will access a list returned from *getSessions* or *getPrunedSessions*.
    
    Parameters
    --------------
    sub_id : int
        A subject id in [0, 33] range
        
    Returns
    ----------
    int
        The index for access a list returned from *getSessions* or *getPrunedSessions*.
        
    Examples
    -------------
    >>> sessions = getSessions('path-to-eremus-data-dir')
    >>> # get the other session of subject 15
    >>> sessions[sub_ot(15)]
    """
    return sub(sub_id, True)


#=============================================================
# GET DATASET LABELS, AND SELECT EMOTIONS
#=============================================================
def get_dataset_labels(data, select_second_label=False):
    """
    Get a list of emotion labels, given a DataFrame containing dataset entries.
    
    Parameters
    -----------
    data : pandas.DataFrame
        a dataframe containing all dataset entries. **Tips**: open *eremus_test.xlsx* file with pandas.read_exel in order to return pandas.DataFrame.
    select_second_label : bool
        If True, it selects the second optional label, otherwise the first (default).
        
    Returns
    -------------
    int
        Emotion Identifier
    """
    if select_second_label:
        return [eval(row.gew_2)[0] if(type(row.gew_2)==str) else None for _, row in data.iterrows()]
    return [eval(row.gew_1)[0] for _, row in data.iterrows()]

def select_emotion(labels, emotion_to_select):
    """
    Given a label and a list of labels, returns the indices of samples that have that label
    
    Parameters
    -----------
    labels : list of int
        The list of labels (emotion identifiers or class identifiers).
    emotion_to_select : int
        Emotion Identifier to select. If you are using Emotions in *eremus_test.xlsx* it should be in [0, 22] range.
        
    Returns
    -------------
    list of int
        The list of indices, where you could find the *emotion_to_select*.
    """
    return [i for i, x in enumerate(labels) if x == emotion_to_select]

def get_raw(raw_fname, start, end):
    """
    Wrapper for getting raw data: it checks file extension and call the proper function
    
    Parameters
    ----------
    raw_fname: str
        The filename of input raw eeg
    start: int
        start sample, used to crop raw
    end: int
        end sample, used to crop raw

    Returns
    -------
    Union[mne.io.RawArray, mne.io.RawEEGLAB, mne.io.RawEDF]
        a RawBase object with eeg data
    """
    # Check Data Extension
    raw_extensions = {'.edf', '.set', '.npz'}
    
    # Preprocessed Data
    if raw_fname.endswith('.npz'):
        # Create info object
        ch_names = list(Session.chs.keys())
        ch_types = ['eeg']*len(ch_names)
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=128)
        # Create Raw
        return mne.io.RawArray(np.load(raw_fname)['arr_0'][:, start:end], info, verbose=False)
    
    # Pruned Data
    elif raw_fname.endswith('.set'):
        eeg = mne.io.read_raw_eeglab(raw_fname, preload=True, verbose=False)
        return eeg.crop(tmin=eeg.times[start], tmax=eeg.times[end],  include_tmax=False)
    
    # Original Raw Data
    elif raw_fname.endswith('.edf'):
        eeg = mne.io.read_raw_edf(raw_fname, preload=True, verbose=False)
        return eeg.crop(tmin=eeg.times[start], tmax=eeg.times[end],  include_tmax=False)