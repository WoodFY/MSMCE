import os
import pickle
import numpy as np

from utils.file_utils import get_file_paths
from utils.file_split import (
    split_nsclc_mzml_files,
    split_crlm_files,
    split_rcc_mzml_files,
)
from utils.ms_processor import (
    load_canine_sarcoma_mzml,
    load_nsclc_mzml,
    load_crlm_mzml,
    load_rcc_mzml,
)
from utils.dataset_split import split_dataset
from utils.bin_ms import binning


def save_raw_mass_spec_data_to_pickle(file_path, mz_arrays, intensity_arrays, labels):
    with open(file_path, 'wb') as file:
        pickle.dump((mz_arrays, intensity_arrays, labels), file)


def save_bin_mass_spec_data_to_pickle(file_path, mz_array, intensity_matrix, labels):
    with open(file_path, 'wb') as file:
        pickle.dump((mz_array, intensity_matrix, labels), file)


def load_raw_mass_spec_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        mz_arrays, intensity_arrays, labels = pickle.load(file)
    return mz_arrays, intensity_arrays, labels


def load_bin_mass_spec_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        mz_array, intensity_matrix, labels = pickle.load(file)
    return mz_array, intensity_matrix, labels


def prepare_canine_sarcoma_dataset(args):
    """
    Prepare the Canine Sarcoma dataset.
     - Mass ranges: 100-1600 Da
     - bin size: 0.1 Da
     - dimension: 15000
    """
    canine_sarcoma_file_paths = get_file_paths(os.path.join(args.root_dir, args.dataset_dir), suffix='.mzML')
    mz_arrays, intensity_arrays, labels = load_canine_sarcoma_mzml(
        file_paths=canine_sarcoma_file_paths,
        label_mapping=args.label_mapping,
        num_classes=args.num_classes
    )

    bin_mz_array, bin_intensity_matrix = binning(
        mz_arrays=mz_arrays,
        intensity_arrays=intensity_arrays,
        mz_min=100.0,
        mz_max=1600.0,
        bin_size=args.bin_size
    )

    X_train, y_train, X_test, y_test = split_dataset(
        X=bin_intensity_matrix,
        y=labels,
        train_size=0.9,
        test_size=0.1,
        random_seed=args.random_seed
    )

    bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, bin_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, bin_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_nsclc_dataset(args):
    """
    Prepare the NSCLC dataset.
     - Mass ranges: 400-1600 Da
     - bin size: 0.1 Da
     - rt binning window: 10 Sec
     - dimension: 12000
    """
    nsclc_file_paths = get_file_paths(os.path.join(args.root_dir, args.dataset_dir), suffix='.mzML')

    train_file_paths, test_file_paths = split_nsclc_mzml_files(file_paths=nsclc_file_paths)

    train_mz_array, X_train, y_train = load_nsclc_mzml(
        file_paths=train_file_paths,
        label_mapping=args.label_mapping,
        mz_min=400.0,
        mz_max=1600.0,
        bin_size=args.bin_size,
        rt_binning_window=args.rt_binning_window
    )
    test_mz_array, X_test, y_test = load_nsclc_mzml(
        file_paths=test_file_paths,
        label_mapping=args.label_mapping,
        mz_min=400.0,
        mz_max=1600.0,
        bin_size=args.bin_size,
        rt_binning_window=args.rt_binning_window
    )

    bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, train_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, test_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_crlm_dataset(args):
    """
    Prepare the CRLM dataset.
     - Mass ranges: 400-1600 Da
     - bin size: 0.1 Da
     - rt binning window: 10 Sec
     - dimension: 12000
    """
    crlm_file_paths = get_file_paths(os.path.join(args.root_dir, args.dataset_dir), suffix='.mzML')

    train_file_paths, test_file_paths = split_crlm_files(file_paths=crlm_file_paths)

    train_mz_array, X_train, y_train = load_crlm_mzml(
        file_paths=train_file_paths,
        label_mapping=args.label_mapping,
        mz_min=400.0,
        mz_max=1600.0,
        bin_size=args.bin_size,
        rt_binning_window=args.rt_binning_window
    )
    test_mz_array, X_test, y_test = load_crlm_mzml(
        file_paths=test_file_paths,
        label_mapping=args.label_mapping,
        mz_min=400.0,
        mz_max=1600.0,
        bin_size=args.bin_size,
        rt_binning_window=args.rt_binning_window
    )

    bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, train_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, test_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_rcc_dataset(args):
    """
    Prepare the RCC dataset.
     - Mass ranges: 70-1060 Da
     - bin size: 0.1 Da
     - rt binning window: 10 Sec
     - dimension: 19000
    """
    rcc_file_paths = get_file_paths(os.path.join(args.root_dir, args.dataset_dir), suffix='.mzML')

    train_file_paths, test_file_paths = split_rcc_mzml_files(
        file_paths=rcc_file_paths,
        train_size=0.9,
        test_size=0.1,
        random_seed=args.random_seed
    )

    train_mz_array, X_train, y_train = load_rcc_mzml(
        file_paths=train_file_paths,
        label_mapping=args.label_mapping,
        mz_min=70.0,
        mz_max=1060.0,
        bin_size=args.bin_size,
        rt_binning_window=args.rt_binning_window
    )
    test_mz_array, X_test, y_test = load_rcc_mzml(
        file_paths=test_file_paths,
        label_mapping=args.label_mapping,
        mz_min=70.0,
        mz_max=1060.0,
        bin_size=args.bin_size,
        rt_binning_window=args.rt_binning_window
    )

    bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, train_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, test_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def get_bin_dataset_path(args):

    if args.dataset_name in ['canine_sarcoma_posion']:
        bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

        if not os.path.exists(bin_dataset_dir):
            os.makedirs(bin_dataset_dir)

        saved_bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_train.pkl"
        saved_bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_test.pkl"
    elif args.dataset_name in ['rcc_posion', 'nsclc', 'crlm']:
        bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

        if not os.path.exists(bin_dataset_dir):
            os.makedirs(bin_dataset_dir)

        saved_bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_train.pkl"
        saved_bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_rt_binning_window_{args.rt_binning_window}_test.pkl"
    else:
        raise ValueError(f'Unknown dataset: {args.datase_name}')

    return saved_bin_train_dataset_path, saved_bin_test_dataset_path


def prepare_dataset(args):

    saved_bin_train_dataset_path, saved_bin_test_dataset_path = get_bin_dataset_path(args)

    if os.path.exists(saved_bin_train_dataset_path) and os.path.exists(saved_bin_test_dataset_path):
        print(f'Loaded data from {saved_bin_train_dataset_path}, {saved_bin_test_dataset_path}')

        train_mz_array, train_intensity_matrix, train_labels = load_bin_mass_spec_data_from_pickle(saved_bin_train_dataset_path)
        test_mz_array, test_intensity_matrix, test_labels = load_bin_mass_spec_data_from_pickle(saved_bin_test_dataset_path)

        X_train, y_train = np.array(train_intensity_matrix), np.array(train_labels)
        X_test, y_test = np.array(test_intensity_matrix), np.array(test_labels)

        print(f'X_train.shape: {X_train.shape} y_train.shape: {y_train.shape}')
        print(f'X_test.shape: {X_test.shape} y_test.shape: {y_test.shape}')

        return X_train, y_train, X_test, y_test
    else:
        print(f'Loaded data from scratch.')

        if args.dataset_name == 'canine_sarcoma_posion':
            X_train, y_train, X_test, y_test = prepare_canine_sarcoma_dataset(args)
        elif args.dataset_name == 'nsclc':
            X_train, y_train, X_test, y_test = prepare_nsclc_dataset(args)
        elif args.dataset_name == 'crlm':
            X_train, y_train, X_test, y_test = prepare_crlm_dataset(args)
        elif args.dataset_name == 'rcc_posion':
            X_train, y_train, X_test, y_test = prepare_rcc_dataset(args)
        else:
            raise ValueError(f'Unknown dataset: {args.dataset_name}')

        print(f'X_train.shape: {X_train.shape} y_train.shape: {y_train.shape}')
        print(f'X_test.shape: {X_test.shape} y_test.shape: {y_test.shape}')

        return X_train, y_train, X_test, y_test


