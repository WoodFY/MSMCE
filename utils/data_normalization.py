import numpy as np


def tic_normalization(data):
    """Total Ion Current (TIC) normalization."""
    # Calculate the total intensity for each sample
    total_intensity = np.sum(data, axis=1, keepdims=True)
    # Avoid division by zero by setting zero total intensity to 1
    total_intensity[total_intensity == 0] = 1
    # Normalize the intensity values
    return data / total_intensity