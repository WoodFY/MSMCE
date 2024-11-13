import os

from utils.file_split import (
    split_nsclc_mzml_files,
    split_crlm_files,
    split_rcc_mzml_files
)
from utils.data_loader import (
    get_file_paths,
    save_bin_mass_spec_data_to_pickle,
    load_canine_sarcoma_mzml,
    load_microorganisms_mzml,
    load_nsclc_mzml,
    load_crlm_mzml,
    load_rcc_mzml
)
from utils.dataset_split import split_dataset
from utils.data_process import bin_spectra


def prepare_canine_sarcoma_dataset(exp_args, label_mapping):
    """
    Prepare the Canine Sarcoma dataset.
     - Mass ranges: 100-1600 Da
     - bin size: 0.1 Da
     - dimension: 15000
    """
    canine_sarcoma_file_paths = get_file_paths(exp_args['root_dir'], exp_args['dataset_dir'], suffix='mzML')
    mz_arrays, intensity_arrays, labels = load_canine_sarcoma_mzml(
        file_paths=canine_sarcoma_file_paths,
        label_mapping=label_mapping,
        num_classes=exp_args['num_classes']
    )

    bin_mz_array, bin_intensity_matrix = bin_spectra(
        mz_arrays=mz_arrays,
        intensity_arrays=intensity_arrays,
        min_mz=100.0,
        max_mz=1600.0,
        bin_size=exp_args['bin_size']
    )

    X_train, y_train, X_test, y_test = split_dataset(
        X=bin_intensity_matrix,
        y=labels,
        train_size=0.9,
        test_size=0.1,
        random_seed=exp_args['random_seed']
    )

    bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, bin_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, bin_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_microorganisms_dataset(exp_args, label_mapping):
    """
    Prepare the Canine Sarcoma dataset.
     - Mass ranges: 100-2000 Da
     - bin size: 0.1 Da
     - dimension: 19000
    """
    microorganisms_file_paths = get_file_paths(exp_args['root_dir'], exp_args['dataset_dir'], suffix='mzML')
    mz_arrays, intensity_arrays, labels = load_microorganisms_mzml(
        file_paths=microorganisms_file_paths,
        label_mapping=label_mapping,
        num_classes=exp_args['num_classes']
    )

    bin_mz_array, bin_intensity_matrix = bin_spectra(
        mz_arrays=mz_arrays,
        intensity_arrays=intensity_arrays,
        min_mz=100.0,
        max_mz=2000.0,
        bin_size=exp_args['bin_size']
    )

    X_train, y_train, X_test, y_test = split_dataset(
        X=bin_intensity_matrix,
        y=labels,
        train_size=0.8,
        test_size=0.2,
        random_seed=exp_args['random_seed']
    )

    bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, bin_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, bin_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_nsclc_dataset(exp_args, label_mapping):
    """
    Prepare the NSCLC dataset.
     - Mass ranges: 400-1600 Da
     - bin size: 0.1 Da
     - rt binning window: 10 Sec
     - dimension: 12000
    """
    nsclc_file_paths = get_file_paths(exp_args['root_dir'], exp_args['dataset_dir'], suffix='mzML')

    train_file_paths, test_file_paths = split_nsclc_mzml_files(file_paths=nsclc_file_paths)

    train_mz_array, X_train, y_train = load_nsclc_mzml(
        file_paths=train_file_paths,
        label_mapping=label_mapping,
        min_mz=400.0,
        max_mz=1600.0,
        bin_size=exp_args['bin_size'],
        rt_binning_window=exp_args['rt_binning_window']
    )
    test_mz_array, X_test, y_test = load_nsclc_mzml(
        file_paths=test_file_paths,
        label_mapping=label_mapping,
        min_mz=400.0,
        max_mz=1600.0,
        bin_size=exp_args['bin_size'],
        rt_binning_window=exp_args['rt_binning_window']
    )

    bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, train_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, test_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_crlm_dataset(exp_args, label_mapping):
    """
    Prepare the CRLM dataset.
     - Mass ranges: 400-1600 Da
     - bin size: 0.1 Da
     - rt binning window: 10 Sec
     - dimension: 12000
    """
    crlm_file_paths = get_file_paths(exp_args['root_dir'], exp_args['dataset_dir'], suffix='mzML')

    train_file_paths, test_file_paths = split_crlm_files(file_paths=crlm_file_paths)

    train_mz_array, X_train, y_train = load_crlm_mzml(
        file_paths=train_file_paths,
        label_mapping=label_mapping,
        min_mz=400.0,
        max_mz=1600.0,
        bin_size=exp_args['bin_size'],
        rt_binning_window=exp_args['rt_binning_window']
    )
    test_mz_array, X_test, y_test = load_crlm_mzml(
        file_paths=test_file_paths,
        label_mapping=label_mapping,
        min_mz=400.0,
        max_mz=1600.0,
        bin_size=exp_args['bin_size'],
        rt_binning_window=exp_args['rt_binning_window']
    )

    bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, train_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, test_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


def prepare_rcc_dataset(exp_args, label_mapping):
    """
    Prepare the RCC dataset.
     - Mass ranges: 70-1060 Da
     - bin size: 0.1 Da
     - rt binning window: 10 Sec
     - dimension: 19000
    """
    rcc_file_paths = get_file_paths(exp_args['root_dir'], exp_args['dataset_dir'], suffix='mzML')

    train_file_paths, test_file_paths = split_rcc_mzml_files(
        file_paths=rcc_file_paths,
        train_size=0.9,
        test_size=0.1,
        random_seed=exp_args['random_seed']
    )

    train_mz_array, X_train, y_train = load_rcc_mzml(
        file_paths=train_file_paths,
        label_mapping=label_mapping,
        min_mz=70.0,
        max_mz=1060.0,
        bin_size=exp_args['bin_size'],
        rt_binning_window=exp_args['rt_binning_window']
    )
    test_mz_array, X_test, y_test = load_rcc_mzml(
        file_paths=test_file_paths,
        label_mapping=label_mapping,
        min_mz=70.0,
        max_mz=1060.0,
        bin_size=exp_args['bin_size'],
        rt_binning_window=exp_args['rt_binning_window']
    )

    bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

    if os.path.exists(bin_dataset_dir) is False:
        os.makedirs(bin_dataset_dir)

    bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_train.pkl"
    bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_test.pkl"
    save_bin_mass_spec_data_to_pickle(bin_train_dataset_path, train_mz_array, X_train, y_train)
    save_bin_mass_spec_data_to_pickle(bin_test_dataset_path, test_mz_array, X_test, y_test)

    return X_train, y_train, X_test, y_test


