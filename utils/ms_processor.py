import os
import numpy as np
from pyteomics import mzml

from utils.bin_ms import binning


def read_mzml(file_path, tic_threshold=None, non_zero_threshold=None, tic_normalize=False, min_max_normalize=False):

    mz_arrays = []
    intensity_arrays = []

    with mzml.read(file_path) as reader:
        spectra = list(reader)

    spectra.sort(key=lambda s: s['scanList']['scan'][0]['scan start time'])

    for spectrum in spectra:
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']

        if tic_threshold is not None:
            # Total Ion Count (TIC) threshold
            tic = np.sum(intensity_array)

            if tic > tic_threshold:

                if tic_normalize:
                    intensity_array = intensity_array / tic
                elif min_max_normalize:
                    intensity_array = (intensity_array - intensity_array.min()) / (intensity_array.max() - intensity_array.min())
                elif tic_normalize and min_max_normalize:
                    raise ValueError("Both tic_normalize and min_max_normalize cannot be True.")

                mz_arrays.append(mz_array)
                intensity_arrays.append(intensity_array)

        elif non_zero_threshold is not None:
            non_zero_intensity_count = np.sum(intensity_array != 0)

            if non_zero_intensity_count > non_zero_threshold:

                if tic_normalize:
                    intensity_array = intensity_array / np.sum(intensity_array)
                elif min_max_normalize:
                    intensity_array = (intensity_array - intensity_array.min()) / (intensity_array.max() - intensity_array.min())
                elif tic_normalize and min_max_normalize:
                    raise ValueError("Both tic_normalize and min_max_normalize cannot be True.")

                mz_arrays.append(mz_array)
                intensity_arrays.append(intensity_array)

        elif tic_threshold is not None and non_zero_threshold is not None:
            raise ValueError("Both tic_threshold and non_zero_threshold cannot")

        elif tic_threshold is None and non_zero_threshold is None:
            raise ValueError("Either tic_threshold or non_zero_threshold must be provided.")

    return mz_arrays, intensity_arrays


def load_mass_spectra_process_to_bin(file_paths, file_type, get_label_function, label_mapping, mz_min, mz_max, bin_size, tic_threshold=None, rt_binning_window=None):

    bin_mz_array = np.arange(mz_min, mz_max, bin_size)
    bin_merged_intensity_matrix = []
    labels = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        if file_type == 'mzML':
            with mzml.read(file_path) as reader:
                spectra = list(reader)
        elif file_type == 'mzXML':
            with mzml.read(file_path) as reader:
                spectra = list(reader)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        spectra.sort(key=lambda s: s['scanList']['scan'][0]['scan start time'])

        label = get_label_function(file_path)
        mapped_label = label_mapping[label]

        mz_arrays = []
        intensity_arrays = []
        rt_list = []

        for spectrum in spectra:
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            rt_seconds = spectrum['scanList']['scan'][0]['scan start time'] * 60

            if tic_threshold:
                tic = np.sum(intensity_array)
                if tic < tic_threshold:
                    continue

            mz_arrays.append(mz_array)
            intensity_arrays.append(intensity_array)
            rt_list.append(rt_seconds)

        _, bin_intensity_matrix = binning(
            mz_arrays=mz_arrays,
            intensity_arrays=intensity_arrays,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_size=bin_size
        )

        if rt_binning_window:
            print(f"Merge binned spectra by RT window of {rt_binning_window} seconds.")
            # Initialize variables for RT window binning
            current_window_bin_intensity_arrays = []
            current_rt_start = 0.0

            # Iterate over the RTs and corresponding binned intensity arrays
            for rt, bin_intensity_array in zip(rt_list, bin_intensity_matrix):
                # If the current RT exceeds the current window, merge the existing window data
                if rt >= current_rt_start + rt_binning_window:
                    # Merge the binned intensity arrays within the current window if there is data
                    if current_window_bin_intensity_arrays:
                        merged_intensity_array = np.mean(current_window_bin_intensity_arrays, axis=0)
                        bin_merged_intensity_matrix.append(merged_intensity_array)
                        labels.append(mapped_label)

                    # Move to the next RT window
                    current_window_bin_intensity_arrays = []
                    current_rt_start += rt_binning_window

                # Add the current binned intensity array to the window's list
                current_window_bin_intensity_arrays.append(bin_intensity_array)

            # Merge the last window if it contains data
            if current_window_bin_intensity_arrays:
                merged_intensity_array = np.mean(current_window_bin_intensity_arrays, axis=0)
                bin_merged_intensity_matrix.append(merged_intensity_array)
                labels.append(mapped_label)

    return bin_mz_array, np.array(bin_merged_intensity_matrix), np.array(labels)


def load_canine_sarcoma_mzml(file_paths, label_mapping, num_classes):

    # classes 2
    if num_classes == 2:
        def get_label_from_path(file_path):
            dir_name = os.path.basename(os.path.dirname(file_path))

            if dir_name == 'Healthy':
                return 'Healthy'
            elif dir_name in ['Myxosarcoma', 'Fibrosarcoma', 'Hemangiopericytoma', 'Malignant peripheral nerve tumor',
                              'Osteosarcoma', 'Undifferentiated pleomorphic', 'Rhabdomyosarcoma', 'Splenic fibrohistiocytic nodules',
                              'Histiocytic sarcoma', 'Soft tissue sarcoma', 'Gastrointestinal stromal sarcoma']:
                return 'Cancerous'
            else:
                raise ValueError(f"Unknown label in file path: {file_path}")
    # classes 12
    elif num_classes == 12:
        def get_label_from_path(file_path):
            dir_name = os.path.basename(os.path.dirname(file_path))

            if dir_name in ['Healthy', 'Myxosarcoma', 'Fibrosarcoma', 'Hemangiopericytoma', 'Malignant peripheral nerve tumor',
                            'Osteosarcoma', 'Undifferentiated pleomorphic', 'Rhabdomyosarcoma', 'Splenic fibrohistiocytic nodules',
                            'Histiocytic sarcoma', 'Soft tissue sarcoma', 'Gastrointestinal stromal sarcoma']:
                return dir_name
            else:
                raise ValueError(f"Unknown label in file path: {file_path}")
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")

    total_mz_arrays = []
    total_intensity_arrays = []
    labels = []

    for file_path in file_paths:
        label = get_label_from_path(file_path)
        mapped_label = label_mapping[label]

        mz_arrays, intensity_arrays = read_mzml(file_path, tic_threshold=1e4)

        total_mz_arrays.extend(mz_arrays)
        total_intensity_arrays.extend(intensity_arrays)
        labels.extend([mapped_label] * len(mz_arrays))

    return total_mz_arrays, total_intensity_arrays, labels


def load_nsclc_mzml(file_paths, label_mapping, mz_min, mz_max, bin_size, rt_binning_window):

    def get_label_from_path(file_path):
        filename = os.path.basename(file_path)

        if 'Xeno092' in filename or 'Xeno441' in filename:
            return 'ADC'
        elif 'Xeno561' in filename or 'Xeno691' in filename:
            return 'SCC'
        else:
            raise ValueError(f"Unknown label in file name: {filename}")

    return load_mass_spectra_process_to_bin(
        file_paths=file_paths,
        file_type='mzML',
        get_label_function=get_label_from_path,
        label_mapping=label_mapping,
        mz_min=mz_min,
        mz_max=mz_max,
        bin_size=bin_size,
        rt_binning_window=rt_binning_window
    )


def load_crlm_mzml(file_paths, label_mapping, mz_min, mz_max, bin_size, rt_binning_window):

    def get_label_from_path(file_path):
        dir_name = os.path.basename(os.path.dirname(file_path))

        if dir_name in ['Control', 'CRLM']:
            return dir_name
        else:
            raise ValueError(f'Unknown label in file path: {file_path}')

    return load_mass_spectra_process_to_bin(
        file_paths=file_paths,
        file_type='mzML',
        get_label_function=get_label_from_path,
        label_mapping=label_mapping,
        mz_min=mz_min,
        mz_max=mz_max,
        bin_size=bin_size,
        rt_binning_window=rt_binning_window
    )


def load_rcc_mzml(file_paths, label_mapping, mz_min, mz_max, bin_size, rt_binning_window):

    def get_label_from_path(file_path):
        dir_name = os.path.basename(os.path.dirname(file_path))

        if dir_name in ['Control', 'RCC']:
            return dir_name
        else:
            raise ValueError(f'Unknown label in file path: {file_path}')

    return load_mass_spectra_process_to_bin(
        file_paths=file_paths,
        file_type='mzML',
        get_label_function=get_label_from_path,
        label_mapping=label_mapping,
        mz_min=mz_min,
        mz_max=mz_max,
        bin_size=bin_size,
        rt_binning_window=rt_binning_window
    )
