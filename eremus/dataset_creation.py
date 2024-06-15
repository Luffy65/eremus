import os
import json
import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None  # default='warn'

# configure eremus Path
configuration_path = Path(__file__).parent
with open(configuration_path / 'configuration.txt') as json_file:
    configuration = json.load(json_file)
    path_to_eremus_data = configuration['path_to_eremus_data']
dataset_file = path_to_eremus_data + 'eremus_test.xlsx'

def sliding_window(path_to_eremus=dataset_file, output_path=path_to_eremus_data+'augmented_eremus.xlsx', s_freq=128, window_size=10, step_size=3):
    """
    Apply Sliding Window to EREMUS Dataset, thus producing data augmentation. Each EREMUS sample is divided into *N* windows, of size *window_size* and whose start points are *step_size* distant. This function read the file at *path_to_eremus* and apply sliding window, thus writing a new file with augmented samples.
    
    Parameters
    ------------
    path_to_eremus : str
        The path to eremus_text.xlsx file. Default value is retrieved from *configuration.txt*. **Tips**: configure the path in *configuration.txt* file, and ignore this parameter (use the default value).
    output_path : str
        The path and filename of the output xlsx file. Default value of output directory is retrieved from *configuration.txt*. Default filename is *augmented_eremus.xlsx*. **Tips**: configure the path in *configuration.txt* file, and ignore this parameter (use the default value).
    s_freq : int
        The sampling frequency used in EEG data in Hz. Default to 128.
    window_size : int
        The size (in seconds) of the time-window used in sliding window algorithm. Default value is 10.
    step_size : int
        The size (in seconds) of the distance between two windows' start points used in sliding window algorithm. Default value is 3.
    """
    # read original dataset
    samples = pd.read_excel(dataset_file)
    # delete additional column
    del samples['Unnamed: 0']

    # get a sample
    sample = samples.iloc[0]

    # create empty dataframe
    new_df = pd.DataFrame()
    # initialize dataframe to fit samples dtypes
    new_df['original_index'] = pd.Series([], dtype=int)
    for k, v in sample.items():
        if k in ['subject_id', 'start_index', 'end_index']:
            new_df[k] = pd.Series([], dtype=int)
        else:
            new_df[k] = pd.Series([], dtype=str)

    # iterate samples
    for i, sample in samples.iterrows():

        # divide sample into sub_sumples of window_size
        for start_index in range(sample.start_index, sample.end_index-window_size*s_freq, step_size*s_freq):
            # compute end_index
            end_index = start_index + s_freq*window_size
            # update start and stop values
            sample.loc['start_index'] = start_index
            sample.loc['end_index'] = end_index
            # add original index
            sample.loc['original_index'] = i
            # add sample to dataframe
            new_df = new_df.append(sample)

    new_df.to_excel(output_path)
    
def get_subject_dataset(subject_id, xls_file=dataset_file, delete_neutral=True, delete_different=True):
    """
    Select from the main dataset only samples that come from a single subject, specified by the *subject_id* 
    
    Parameters
    ------------
    subject_id : int
        The subject identifier. All samples in output will belong to this subject.
    xls_file : str
        The path to *eremus_text.xlsx file*. Default value is retrieved from *configuration.txt*. **Tips**: configure the path in *configuration.txt* file, and ignore this parameter (use the default value).
    delete_neutral : bool
        If True, samples that belong to the neutral class are discarded.
    delete_different : bool
        If True, samples that belong to the *different* class are discarded.
    
    Returns
    --------------
    pandas.DataFrame
        A DataFrame containing all the selected samples.
    """
    # open xls file
    df = pd.read_excel(xls_file)
    # select subject
    df = df[df.subject_id == subject_id]
    # discard neutral and/or different emotions
    if delete_neutral and delete_different:
        df = df[pd.eval(df.gew_1)[:, 0] < 20]
    elif delete_neutral:
        df = df[pd.eval(df.gew_1)[:, 0] != 20]
    elif delete_different:
        df = df[pd.eval(df.gew_1)[:, 0] != 21]
    df.reset_index(inplace=True, drop=True)
    df.rename(columns = {'Unnamed: 0': 'original_index'}, inplace = True)
    return df

def create_single_subject_datasets(xls_file=dataset_file, w_dir=path_to_eremus_data+"single_subject\\"):
    """
    Create single subject datasets from the main dataset. 
    
    Parameters
    ------------
    xls_file : str
        The path to *eremus_text.xlsx file*. Default value is retrieved from *configuration.txt*. **Tips**: configure the path in *configuration.txt* file, and ignore this parameter (use the default value).
    w_dir : str
        The output directory in which all single subject dataset will be saved in xlsx format. All output file will be named: *subject_<SUBJECT_ID>.xlsx* 
    """
    os.makedirs(w_dir, exist_ok=True)
    for subject_id in range(pd.read_excel(xls_file).subject_id.max() + 1):
        # Get Sub-Dataset for a Single subject
        df = get_subject_dataset(subject_id, xls_file, delete_neutral=True, delete_different=True)
        file_name = w_dir + 'subject_' + str(subject_id) + '.xlsx' 
        df.to_excel(file_name)