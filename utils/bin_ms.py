import numpy as np


def binning(mz_arrays, intensity_arrays, mz_min, mz_max, bin_size=0.1):
    num_bins = int((mz_max - mz_min) / bin_size)
    bin_mz_array = np.arange(mz_min, mz_max, bin_size)
    bin_intensity_matrix = []

    for mz_array, intensity_array in zip(mz_arrays, intensity_arrays):
        assert len(mz_array) == len(intensity_array), f'Length of mz and intensity should be the same!'

        bin_intensity_array = np.zeros(num_bins)
        for mz, intensity in zip(mz_array, intensity_array):
            if mz_min <= mz < mz_max:
                bin_index = int((mz - mz_min) / bin_size)
                bin_intensity_array[bin_index] += intensity

        bin_intensity_matrix.append(bin_intensity_array)

    return bin_mz_array, bin_intensity_matrix


