#!/usr/bin/env python
# coding: utf-8

# ## Spatial Filtering

import mne
import json
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms, utils

# we need montage for calculating distances beetween sensors

# get a sample from dataset
configuration_path = Path(__file__).parent.parent
with open(configuration_path /'configuration.txt') as json_file:
    configuration = json.load(json_file)
    path_to_eremus_data = Path(configuration['path_to_eremus_data'])
samples = pd.read_excel(path_to_eremus_data/'eremus_test.xlsx')
pruned_eeg_root_dir = path_to_eremus_data/Path('recordings_pruned_with_ICA')
eeg_file = samples.iloc[0]['filename_pruned']
sample = mne.io.read_raw_eeglab(pruned_eeg_root_dir/eeg_file, verbose = False)

# get and plot montage
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
sample.set_montage(ten_twenty_montage)
montage = sample.get_montage()
#fig = montage.plot()

# channel names for EREMUS dataset
ch_names = sample.ch_names

# create a dict that maps dig points indices to channel names
ch_dict = {k+3: v for k, v in enumerate(ch_names)}
# create a dict that maps channel names to dig points indices
ch_dict_r = {k: v+3 for v, k in enumerate(ch_names)}

# extract positions
positions = [montage.dig[ch_dict_r[c]]['r'] for c in ch_names]



def get_nn_for_channel(channel, nn_max_index=7, verbose=False):
    """
    Calculate the N = *nn_max_index* nearest neighbors of the selected channel.
    
    Parameters
    -------------
    channel : str
        Target channel for distance calculus. It must be provided with channel name (i.e 'Cz', 'PO10', etc.)
    nn_max_index : int 
        The number of nearest neighbors to find.
    verbose : bool
        If True, print information for debug purpose.
    
    Returns
    -----------
    list[int] 
        positions of the N = *nn_max_index* nearest neighbors of the selected channel.
    """
    if channel not in ch_names:
        raise ValueError("Please provide a valid channel name. You could find allowed values in spatial_filter.ch_names list")
    # calculate distances from channel
    dist_from_channel = []
    for c in ch_names:
        dist_from_channel.append(np.linalg.norm(positions[ch_dict_r[channel] - 3] - positions[ch_dict_r[c] - 3]))
    # sort distances, preserving indices of channels
    nn = sorted(enumerate(dist_from_channel), key = lambda x : x[1])
    
    # for debug - set verbose to True
    if verbose:
        for i in range(0, nn_max_index):
            # access channel names from dig indices using nn[i][0]+3
            print(ch_dict[nn[i][0]+3], end='\t')
            # access channel names from ch_names using nn[i][0]
            print(ch_names[nn[i][0]], end='\t')
            # access distance from channel
            print(nn[i][1])
    
    # extract and return the first nn_max_index positions
    nn_positions = [nn[i][0] for i in range(len(nn))]
    # extract and return the first nn_max_index distances
    nn_distances = [nn[i][1] for i in range(len(nn))]
    
    return nn_positions[:nn_max_index], nn_distances[:nn_max_index]

# iter the last function for each target channel
def nn(nn_max_index=7):
    """
    Calculate the N = *nn_max_index* nearest neighbors for all channels.
    
    Parameters
    --------------
    nn_max_index : int 
        The number of nearest neighbors to find.
    verbose : bool
        If True, print information for debug purpose.
    
    Returns
    ---------------
    dict
        Supposing you are saving the return value in nn_dict:
        
            - nn_dict['ch_name'][i] returns the name of the i_th target channel
            - nn_dict['ch_pos'][i] returns the position of the i_th target channel
            - nn_dict['nn_names'][i] returns the names of the N = nn_max_index nearest neighbors for target channel
            - nn_dict['nn_pos'][i] returns the positions of the N = nn_max_index nearest neighbors for target channel
            - nn_dict['nn_dist'][i] returns the distances of the N = nn_max_index nearest neighbors for target channel
    """
    
    # create the empty dict
    nn_dict = {
        'ch_name': [],
        'ch_pos': [],
        'nn_names': [],
        'nn_pos': [],
        'nn_dist': []
    }
    # iter all target channels
    for pos, channel in enumerate(ch_names):
        # calculate positions of nearest neighbors
        nn_pos, nn_dist = get_nn_for_channel(channel, nn_max_index=nn_max_index)
        # calculate names of nearest neighbors
        nn_names = [ch_dict[i+3] for i in nn_pos]
        # add entry to dict
        nn_dict['ch_name'].append(channel)
        nn_dict['ch_pos'].append(pos)
        nn_dict['nn_names'].append(nn_names)
        nn_dict['nn_pos'].append(nn_pos)
        nn_dict['nn_dist'].append(nn_dist)
        
    return nn_dict

# calculate dictionary of nn for each channel
nn_dict = nn(nn_max_index=32)

# convert to dataframe
nn_df = pd.DataFrame(nn_dict)

# extract distances for all sensors
dist = np.array([np.array(a) for a in nn_df['nn_dist']])

# provide weigths in function of distances
def get_weights(distance):
    """
    Given a distance (or an array of distances), calculate the corresponding weigth to associate in average calculus. 
    The weight is proportional to the inverse of the distance. 
    If distance = 0 (distance of sensors from themselves) weight is 1.
    If distance = dist.max() (max distance beetween two sensor locations in the 10-20 system) weight is 0.
    
    Parameters
    ------------
    distance : Union[float, numpy.ndarray] 
        Single distance or Array of distances.
    Returns
    ------------
    Union[float, numpy.ndarray] 
        The corresponding weigth/s that associate with distance/s in average calculus.
    """
    
    global dist
    maxim = dist.max()
    return 1 - distance/maxim


# In[7]:


def spatial_filter(raw, N=7, weighted=True, return_only_data=False):
    """
    Calculate the Spatial filter for a given Raw (or for a given numpy array of shape *(N_CHAN, N_TIME_POINTS)*).
    Values of each channel are averaged with those of the first N nearest neighbors.
    Mean may be simple or weigthed in function of distance.
    
    Parameters
    --------------
    raw : Union[mne.io.RawXXX, numpy.ndarray]
        EEG raw data to filter.
    N : int
        The number of nearest neighbors involved in averaging.
    weighted : bool
        If True, select the type of mean to *weighted*.
    return_only_data : bool
        If True, it always returns numpy.ndarray, i.e. the filtered data are not wrapped in a Raw object.
        If False, and *raw* is a Raw object, it returns a new Raw object with filtered data.
    
    Returns
    ------------------
    Union[mne.io.RawXXX, numpy.ndarray]
        The filtered data. If *return_only_data* is False and *raw* is Raw, it returns a new Raw object with filtered data, otherwise an array containing filtered data (not wrapped in Raw object).
        
    References
    ------------
    Michel Christoph M., Brunet Denis
    EEG Source Imaging: A Practical Review of the Analysis Steps  
    in "Frontiers in Neurology"   
    n.10, 2019
    https://www.frontiersin.org/article/10.3389/fneur.2019.00325     
    doi: 10.3389/fneur.2019.00325      
    """
    # check N
    assert N<=32, "N must be in range(num_channels), number of chans is 32 in EREMUS"
    # check raw type
    is_raw_numpy = isinstance(raw, np.ndarray)
    # get data
    raw_data = raw if is_raw_numpy else raw.get_data()
    # create the new_data container
    new_raw_data = raw_data
    # interate position and name of channels
    for pos, channel in enumerate(ch_names):
        # get position of nn for channel
        nn_pos = nn_df.iloc[pos].nn_pos[0:N]
        # select nn
        nn = raw_data[nn_pos]
        # mask min and max values along sensor ax
        mask = np.logical_or(nn == nn.max(0), nn == nn.min(0))
        nn_masked = np.ma.masked_array(nn, mask = mask)
        if weighted:
            # extract nn distances for channel
            dist = np.array(nn_df.iloc[pos].nn_dist)[0:N]
            # calculate weighted mean, excluding min and max values,update new_raw_data
            new_raw_data[pos] = np.ma.average(nn_masked, axis=0, weights=get_weights(dist))
        else:
            # calculate mean, excluding min and max values,update new_raw_data
            new_raw_data[pos] = nn_masked.mean(0)
        
    if return_only_data or is_raw_numpy:
        return new_raw_data
    else:
        return mne.io.RawArray(new_raw_data, raw.info, verbose=False)

if __name__ == "__main__":
    # ### Explorative Analysis

    # Obiettivo primario dell'analisi esplorativa è quello di determinare per ogni canale quali sono i suoi n canali più vicini.
    # Per farlo procediamo per steps:
    # - analisi di un montaggio 10-20
    # - estrazione delle posizioni dei sensori
    # - determinare la distanza tra un sensore e tutti gli altri
    # - per ogni sensore, determinare gli n sensori con distanza minima
    # 
    # #### Step1
    # Come è fatto un montaggio?
    # 
    # Ogni montaggio è un oggetto DigMontage, che contiene essenzialmente una lista di DigPoint e una lista dei nomi dei canali.
    # nella lista dei dig point sono inclusi SEMPRE tre punti in più rispetto ai canali: LPA, RPA e Nasion. Tali punti non devono  essere presi in considerazione.
    # 
    # #### Step2: estrazione posizioni

    # #### Step 3 e 4
    # 
    # Calcoliamo prima la distanza tra un sensore e i restanti, ordiniamo i sensori per distanza e ricaviamo i 7 più vicini.
    # Generalizziamo in una funzione parametrizzando il canale di partenza e il numero di vicini da trovare.

    # In[8]:


    # calculate distances of sensors from Cz
    dist_from_Cz = []
    for c in ch_names:
        dist_from_Cz.append(np.linalg.norm(positions[ch_dict_r['Cz'] - 3] - positions[ch_dict_r[c] - 3]))
    # sort distances saving indices of sensors
    # nn contains entries of type (CHANNEL, DISTANCE FROM Cz)
    # the first n entries of nn are the n nearest neighbors of Cz
    # in order to access indices of sensors use nn_Cz[i][0]
    # note that dig indices are greater of 3
    nn_Cz = sorted(enumerate(dist_from_Cz), key = lambda x : x[1])
    for i in range(0, 7):
        # access channel names from dig indices using nn_Cz[i][0]+3
        print(ch_dict[nn_Cz[i][0]+3], end='\t')
        # access channel names from ch_names using nn_Cz[i][0]
        print(ch_names[nn_Cz[i][0]], end='\t')
        # access distance from Cz
        print(nn_Cz[i][1])


    # #### Test functions

    # In[9]:


    nn_pos_Cz = get_nn_for_channel('Cz', verbose=True, nn_max_index = 32)


    # ### Implementation

    # Scopo della fase di implementazione è la creazione del filtro spaziale.
    # 
    # I passi da seguire per ogni elettrodo e per ogni istante di tempo sono:
    # 
    # - Determinare il valore dei 6 elettrodi vicini, più l'elettrodo target
    # - ordinare i 7 valori in ordine crescene
    # - scartare i calori di massimo e di minimo
    # - fare una media pesata dei valori in funzione della loro distanza
    # 
    # I vari passaggi verranno fatti prima per un elettrodo di esempio, poi wrappati in una funzione. 
    # Si cercherà di far nascere i calcoli in parallelo x tutti gli istanti di tempo.
    # La funzione verrà creata in due varianti, una con la media semplice, l'altra con la media pesata.

    # In[10]:


    # select channel Cz - calculate positions and names of nn
    nn_pos_Cz, _ = get_nn_for_channel('Cz')
    nn_names_Cz = [ch_dict[i+3] for i in nn_pos_Cz]


    # In[11]:


    # calculate dictionary of nn for each channel
    nn_dict = nn()
    # convert to dataframe
    nn_df = pd.DataFrame(nn_dict)
    # extract positions - verify if it is correct
    nn_pos_Cz == nn_df.iloc[0].nn_pos


    # In[12]:


    # extract data from sample and copy 
    raw = sample.copy()
    raw_data = raw.get_data()
    new_raw_data = raw_data

    # extract nn positions for sensor 0 (Cz)
    pos_0 = nn_df.iloc[0].nn_pos

    # select only Cz's nearest neighbourgs
    nn_0 = raw_data[pos_0]

    # extract nn distances for sensor 0 (Cz) (for weighted version)
    dist_0 = np.array(nn_df.iloc[0].nn_dist)

    # search max and min along sensor ax for each time point and produce a mask
    # if entry is True that position is of a min or of a max
    mask = np.logical_or(nn_0 == nn_0.max(0), nn_0 == nn_0.min(0))
    # discard max and min points
    nn_0_masked = np.ma.masked_array(nn_0, mask = mask)


    # In[13]:


    # calculate mean along sensor ax for each time point (for simple version)
    nn_0_filtered = nn_0_masked.mean(0)
    # substitute new_raw_data
    new_raw_data[0] = nn_0_filtered


    # In[14]:


    # calculate mean along sensor ax for each time point (for weighted version)
    nn_0_filtered = np.ma.average(nn_0_masked, axis=0, weights=get_weights(dist_0))
    # substitute new_raw_data
    new_raw_data[0] = nn_0_filtered


    # In[15]:


    # test simple version
    spatial_filter(sample, weighted=False).get_data()


    # In[16]:


    # test weighted version
    spatial_filter(sample).get_data()