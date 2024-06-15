import mne
import torch
import warnings
import numpy as np
from eremus.preprocessing.spatial_filter import spatial_filter


class FixedCrop(object):
    """Crop the EEG in a sample from a given start point.

    Parameters
    --------------
    output_size : int
        Desired output size (on time axis).
    start : int
        Start point for crop operation. It is relative to the observation i.e. *start* = 0 is the start sample.
    """

    def __init__(self, output_size, start=0):
        assert isinstance(output_size, int)
        assert isinstance(start, int)
        if isinstance(output_size, int):
            self.output_size = output_size 
        if isinstance(start, int):
            self.start = start

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        
        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1] 
        new_d = self.output_size
        if (self.start + self.output_size)>=d:
            raise ValueError("start + output_size exceeds the sample length")

        start = self.start
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'emotion': emotion}
    
class RandomCrop(object):
    """Crop randomly the EEG in a sample.

    Parameters
    --------------
    output_size : int
        Desired output size (on time axis).
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']

        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1]
        new_d = self.output_size
        if new_d>=d:
            raise ValueError("output_size exceeds the sample length")

        start = np.random.randint(0, d - new_d)
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'emotion': emotion}

class PickData(object):
    """Pick only EEG channels.
    """
    
    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        #set montage
        data_chans = eeg.ch_names[4:36]
        eeg.pick_channels(data_chans)
        
        return {'eeg': eeg, 'emotion': emotion}
    
class SetMontage(object):
    """Set 10-20 montage to the Raw object. Use this transform only if the eeg is a mne.RawBase object.
    """
    
    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        eeg.set_montage(ten_twenty_montage)
        
        return {'eeg': eeg, 'emotion': emotion}
    
class ToArray(object):
    """Convert the eeg file in a sample to numpy.ndarray."""

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        # discard times, select only array data
        if not isinstance(eeg, np.ndarray):
            eeg = eeg[:][0]
        else:
            warnings.warn("This operation is unuseful, since your data are alreay in numpy format")
        return {'eeg': eeg, 'emotion': emotion}
    
class ToMatrix(object):
    """
    Convert the eeg file in a sample to numpy.ndarray of shape (9, 9, T).
    eeg must be a mne.RawBase object or a numpy.ndarray of shape (C, T), being C the number of eeg channels and T the number of time-points.
    Channels are spatially encoded into a matrix 9x9 following the RA-CNN directives.
    
    References
    ---------------
    Heng Cui, Aiping Liu, Xu Zhang, Xiang Chen, Kongqiao Wang, Xun Chen, 
    EEG-based emotion recognition using an end-to-end regional-asymmetric convolutional neural network,
    Knowledge-Based Systems,
    Volume 205,
    2020,
    106243,
    ISSN 0950-7051,
    https://doi.org/10.1016/j.knosys.2020.106243.
    """
    
    def __init__(self):
        self.location = {
            'Cz': (4, 4),
            'Fz': (2, 4),
            'Fp1': (0, 3),
            'F7': (2, 0),
            'F3': (2, 2),
            'FC1': (3, 3),
            'C3': (4, 2),
            'FC5': (3, 1),
            'FT9': (3, 0),
            'T7': (4, 0),
            'CP5': (5, 1),
            'CP1': (5, 3),
            'P3': (6, 2),
            'P7': (6, 0),
            'PO9': (7, 0),
            'O1': (8, 3),
            'Pz': (6, 4),
            'Oz': (8, 4),
            'O2': (8, 5),
            'PO10': (7, 8),
            'P8': (6, 8),
            'P4': (6, 6),
            'CP2': (5, 5),
            'CP6': (5, 7),
            'T8': (4, 8),
            'FT10': (3, 8),
            'FC6': (3, 7),
            'C4': (4, 6),
            'FC2': (3, 5),
            'F4': (2, 6),
            'F8': (2, 8),
            'Fp2': (0, 5)
        }
        
        self.ndlocation = {i: loc for i, (_, loc) in enumerate(self.location.items())}

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        
        # Check eeg instance type
        if isinstance(eeg, np.ndarray):
            
            # Eeg is a np.ndarray
            # Create an empty matrix (filled with 0)
            eeg_matrix = np.zeros((9, 9, eeg[:][0].shape[0]))
            # Encode array elements in matrix
            for chan, coords in self.ndlocation.items():
                eeg_matrix[coords][:] = eeg[chan]
        else:
            
            # Eeg is a Raw object
            # Create an empty matrix (filled with 0)
            eeg_matrix = np.zeros((9, 9, eeg[:][0].shape[1]))
            # Encode elements in matrix
            for chan, coords in self.location.items():
                eeg_matrix[coords][:] = eeg[chan][0].reshape(-1)

        return {'eeg': eeg_matrix, 'emotion': emotion}

class ToTensor(object):
    """Convert the eeg in a sample from numpy.ndarray format to torch.Tensor.
    It should be inkoed as last transform.
    
    Parameters
    -------------
    interface : str
        Could be 'dict' or 'unpacked_values'. It changes the output interface.
    eeg_tensor_type : str
        Could be 'float32' or 'foat64'. It changes the tensor type.
    label_interface : str
        Could be 'tensor' or 'long'. It changes the label type.
    """
    
    def __init__(self, interface='dict', eeg_tensor_type = 'float32', label_interface='tensor'):
        self.interfaces = ['dict', 'unpacked_values']
        self.eeg_tensor_types = ['float64', 'float32']
        self.label_interfaces = ['tensor', 'long']
        
        assert isinstance(interface, str)
        if interface not in self.interfaces:
            raise ValueError("interface must be one of " + str(self.interfaces))
        if isinstance(interface, str):
            self.interface = interface
            
        assert isinstance(eeg_tensor_type, str)
        if eeg_tensor_type not in self.eeg_tensor_types:
            raise ValueError("eeg_tensor_type must be one of " + str(self.eeg_tensor_types))
        if isinstance(eeg_tensor_type, str):
            self.eeg_tensor_type = eeg_tensor_type
        
        assert isinstance(label_interface, str)
        if label_interface not in self.label_interfaces:
            raise ValueError("label_interface must be one of " + str(self.label_interfaces))
        if isinstance(label_interface, str):
            self.label_interface = label_interface

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        
        if self.eeg_tensor_type=='float32':
            eeg = eeg.astype(np.float32)
        eeg = torch.from_numpy(eeg)
            
        if self.label_interface=='tensor':
            emotion = torch.LongTensor([emotion])
        
        if self.interface=='dict':
            return {'eeg': eeg, 'emotion': emotion}
        elif self.interface=='unpacked_values':
            return eeg, emotion
        
class SpatialFilter(object):
    """
    Apply Spatial Filter to the given sample.
    
    Parameters
    -------------
    N : int
        The number of nearest neighbors used in spatial filtering.
    weighted : bool
        If True, weights are applied in algorithms, in function of distance from the target sensor.
    
    References
    ------------
    Michel Christoph M., Brunet Denis
    EEG Source Imaging: A Practical Review of the Analysis Steps  
    in "Frontiers in Neurology"   
    n.10, 2019
    https://www.frontiersin.org/article/10.3389/fneur.2019.00325     
    doi: 10.3389/fneur.2019.00325      
    """
    def __init__(self, N=7, weighted=True):
        self.N = N
        self.weighted = weighted

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        
        if isinstance(eeg, torch.Tensor):
            raise NotImplementedError("Spatial Filter is only supported for raw objects on numpy arrays:\n\tPlease insert this Transform before ToTensor")
        elif isinstance(eeg, np.ndarray) and len(eeg.shape)!=2:
            raise ValueError("Spatial Filter is only supported for raw objects on numpy arrays with shape (C, T):\n\tPlease insert this Transform before ToMatrix")
        
        eeg = spatial_filter(eeg, N=self.N, weighted=self.weighted)
        
        return {'eeg': eeg, 'emotion': emotion}

FixedCrop_V2 = FixedCrop 
RandomCrop_V2 = RandomCrop
SetMontage_V2 = SetMontage
ToArray_V2 = ToArray
ToMatrix_V2 = ToMatrix
ToTensor_V2 = ToTensor
SpatialFilter_V2 = SpatialFilter
