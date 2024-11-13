import numpy as np


def bin_spectra(mz_arrays, intensity_arrays, min_mz=100.0, max_mz=1600.0, bin_size=0.1):
    num_bins = int((max_mz - min_mz) / bin_size)
    bin_mz_array = np.arange(min_mz, max_mz, bin_size)
    bin_intensity_matrix = []

    for mz_array, intensity_array in zip(mz_arrays, intensity_arrays):
        assert len(mz_array) == len(intensity_array), f'Length of mz and intensity should be the same!'

        bin_intensity_array = np.zeros(num_bins)
        for mz, intensity in zip(mz_array, intensity_array):
            if min_mz <= mz < max_mz:
                bin_index = int((mz - min_mz) / bin_size)
                bin_intensity_array[bin_index] += intensity

        bin_intensity_matrix.append(bin_intensity_array)

    return bin_mz_array, bin_intensity_matrix


