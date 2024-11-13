import numpy as np


def log_intensity_transformation(data, epsilon=1e-6, base='natural'):
    """
    Apply logarithm transformation to the data.

    Args:
        data (np.ndarray): Input data.
        epsilon (float): A small constant added to each element to avoid computing log(0).
        base (str): The base of the logarithm, which can be 'natural', '10', or '2'.

    Returns:
        np.ndarray: The data after applying the logarithmic transformation.
    """

    data += epsilon

    if base == 'natural':
        return np.log(data)
    elif base == '10':
        return np.log10(data)
    elif base == '2':
        return np.log2(data)
    else:
        raise ValueError(f"Unsupported base of logarithm: {base}")