"""
Configuration file
"""
import numpy as np

# General config
_VERBOSE_LEVEL = 1

# Scripts config
# 8 speeds between (0.8, 1.2); remove the speed with value 1
_SPEEDS = np.delete(np.linspace(0.8, 1.2, 9), 4)

# 8 semitones between (-200, 200); remove the semitone with value 0
_SEMITONES = np.delete(np.linspace(-200, 200, 9), 4)

_NOISES = ['preprocessing/noises/ambiance.wav',
           'preprocessing/noises/crowd.wav',
           'preprocessing/noises/street.wav',
           'preprocessing/noises/driving.wav']


# Core config
_EARLY_STOP_RANGE = None
_DATA_CSV = None


class Config:
    def __init__(self):
        # General config
        self.verbose_level = _VERBOSE_LEVEL
        # Core config
        self.early_stop_range = _EARLY_STOP_RANGE
        self.data_csv = _DATA_CSV
