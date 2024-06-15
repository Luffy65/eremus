# EremusDataset - ExtractedFeatures

import mne
import torch
import numpy as np

class FixedCrop(object):
    """Crop the feature array in a sample from a given start point.
    Feature array is expected to be of shape (C, F, B) being C the number of channels, F te number of features and B the number of frequency bands.

    Parameters
    --------------
    output_size : int
        Desired output size (on feature axis).
    start : int
        Start feature-point for crop operation.
    """

    def __init__(self, output_size, start=0):
        assert isinstance(output_size, int)
        assert isinstance(start, int)
        if isinstance(output_size, int):
            self.output_size = output_size 
        if isinstance(start, int):
            self.start = start

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        C = features.shape[0]
        F = features.shape[1]

        start = self.start
        stop = start + self.output_size
        if stop>=F:
            raise ValueError("start + output_size exceeds the sample length")

        features = features[:, start:stop, :]
        
        return features, emotion
    
class RandomCrop(object):
    """Crop randomly the feature array in a sample.
    Feature array is expected to be of shape (C, F, B) being C the number of channels, F te number of features and B the number of frequency bands.

    Parameters
    --------------
    output_size : int
        Desired output size (on feature axis).
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        C = features.shape[0]
        F = features.shape[1]

        if self.output_size>F:
            raise ValueError("output_size exceeds the sample length: Please provide a crop_size < " + str(F))
        start = np.random.randint(0, F - self.output_size)
        stop = start + self.output_size

        features = features[:, start:stop, :]
        
        return features, emotion
    
class SelectBand(object):
    """Select only specified band.
    Feature array is expected to be of shape (C, F, B) being C the number of channels, F te number of features and B the number of frequency bands. SelectBand produces samples of shape (C, F).

    Parameters
    ---------------
    band : Union[str, int]
        Desired band. It could be one of keys provided in *bands* or a integer identifier in range [0, 4].
    """
    
    bands = {
        'delta': 0,
        'theta': 1,
        'alpha': 2,
        'beta': 3,
        'gamma': 4
    }
    """
    Allowed bands.
    """

    def __init__(self, band):
        if not (isinstance(band, str) or isinstance(band, int)):
            raise TypeError("The band parameter should be a string among " + str(self.bands.keys()) + " or a integer in range [0, 4]")
        if isinstance(band, int):
            if band>4 or band<0:
                raise ValueError('Please provide a band_id in range [0, 4]')
            self.band = band
        elif isinstance(band, str):
            if band not in self.bands.keys():
                raise ValueError('Please provide a valid band name. Valid names are ' + str(self.bands.keys()))
            self.band = self.bands[band]

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        C = features.shape[0]
        F = features.shape[1]

        features = features[:, :, self.band].reshape(C, F)
        
        return features, emotion
    
class ToMatrix(object):
    """
    Convert the features in a sample to numpy.ndarray of shape (9, 9, F, [B]). Feature array is expected to be of shape (C, F, B) or (C, F), being C the number of channels, F te number of features and B the number of frequency bands.
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
        features, emotion = sample[0], sample[1]
        
        # Check features dimensions
        n_dim = len(features.shape)
        if n_dim==2:
            # Features is of shape (C, F)
            # Create an empty matrix (filled with 0)
            f_matrix = np.zeros((9, 9, features.shape[1]))
            # Encode array elements in matrix
            for chan, coords in self.ndlocation.items():
                f_matrix[coords][:] = features[chan]
        elif n_dim==3:
            f_matrix = np.zeros((9, 9, features.shape[1],  features.shape[2]))
            # Encode array elements in matrix
            for chan, coords in self.ndlocation.items():
                f_matrix[coords][:][:] = features[chan]

        return f_matrix, emotion 

class ToTensor(object):
    """Convert the features in a sample from numpy.ndarray format to torch.Tensor.
    It should be inkoed as last transform.
    
    Parameters
    -------------
    tensor_type : type
        Could be numpy.float32 or numpy.float64, or other allowed tensor types. It changes the tensor type.
    """
    
    def __init__(self, tensor_type=np.float32):
        self.tensor_type=tensor_type

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        features = features.astype(self.tensor_type)
        features = torch.from_numpy(features)
            
        return features, emotion