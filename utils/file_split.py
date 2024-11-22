import os

from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_files(file_paths, train_size=0.9, test_size=0.1, random_seed=3407):
    """
    Split the file paths into train and test sets.
    """

    if test_size * len(file_paths) < 1:
        test_size = 1 / len(file_paths)
        train_size = 1 - test_size

    train_file_paths, test_file_paths = train_test_split(
        file_paths,
        train_size=train_size,
        test_size=test_size,
        random_state=random_seed
    )

    return train_file_paths, test_file_paths


def split_files_by_label(file_paths, get_label_function, train_size=0.9, test_size=0.1, random_seed=3407):
    label_to_file_paths = defaultdict(list)

    for file_path in file_paths:
        label = get_label_function(file_path)
        label_to_file_paths[label].append(file_path)

    total_train_file_paths = []
    total_test_file_paths = []

    for label, label_file_paths in label_to_file_paths.items():
        train_file_paths, test_file_paths = split_files(
            label_file_paths,
            train_size=train_size,
            test_size=test_size,
            random_seed=random_seed
        )

        total_train_file_paths.extend(train_file_paths)
        total_test_file_paths.extend(test_file_paths)

    return total_train_file_paths, total_test_file_paths


def split_files_by_label_sample_id(file_paths, get_label_function, get_sample_id_function, train_size=0.9, test_size=0.1, random_seed=3407):
    """
        Split the file paths into train and test sets based on both label and sample id.
        If one file of a sample_id is in the training set, all files of that sample_id
        will be in the training set, and vice versa for the test set.
        """
    # Create a dictionary to group files by label and then by sample_id
    label_to_sample_files = defaultdict(lambda: defaultdict(list))

    for file_path in file_paths:
        label = get_label_function(file_path)
        sample_id = get_sample_id_function(file_path)
        label_to_sample_files[label][sample_id].append(file_path)

    total_train_file_paths = []
    total_test_file_paths = []

    # Iterate over each label group
    for label, sample_files_dict in label_to_sample_files.items():
        sample_ids = list(sample_files_dict.keys())

        if test_size * len(sample_ids) < 1:
            test_size = 1 / len(sample_ids)
            train_size = 1 - test_size

        # Split the sample_ids for the current label into train and test sets
        train_sample_ids, test_sample_ids = train_test_split(
            sample_ids,
            train_size=train_size,
            test_size=test_size,
            random_state=random_seed
        )

        # Assign files to train and test sets based on the split sample_id
        for sample_id in train_sample_ids:
            total_train_file_paths.extend(sample_files_dict[sample_id])

        for sample_id in test_sample_ids:
            total_test_file_paths.extend(sample_files_dict[sample_id])

    return total_train_file_paths, total_test_file_paths


def split_nsclc_mzml_files(file_paths, random_seed=3407):
    total_counts = len(file_paths)

    sample_to_file_paths = defaultdict(list)

    for file_path in file_paths:
        filename = os.path.basename(file_path)

        if 'Xeno092' in filename:
            sample_to_file_paths['Xeno092'].append(file_path)
        elif 'Xeno441' in filename:
            sample_to_file_paths['Xeno441'].append(file_path)
        elif 'Xeno561' in filename:
            sample_to_file_paths['Xeno561'].append(file_path)
        elif 'Xeno691' in filename:
            sample_to_file_paths['Xeno691'].append(file_path)
        else:
            raise ValueError(f'Unknown sample name: {filename}')

    train_file_paths = []
    test_file_paths = []

    for sample, file_paths in sample_to_file_paths.items():
        train_sample_paths, test_sample_paths = split_files(file_paths, random_seed=random_seed)
        train_file_paths.extend(train_sample_paths)
        test_file_paths.extend(test_sample_paths)

    assert total_counts == (len(train_file_paths) + len(test_file_paths)), 'Some files are missing'

    return train_file_paths, test_file_paths


def split_crlm_files(file_paths):

    def get_label_from_path(file_path):
        dir_name = os.path.basename(os.path.dirname(file_path))

        if dir_name in ['Control', 'CRLM']:
            return dir_name
        else:
            raise ValueError(f'Invalid directory name: {dir_name}')

    return split_files_by_label(file_paths, get_label_from_path)


def split_rcc_mzml_files(file_paths, train_size=0.9, test_size=0.1, random_seed=3407):

    def get_label_from_path(file_path):
        dir_name = os.path.basename(os.path.dirname(file_path))

        if dir_name in ['Healthy', 'RCC']:
            return dir_name
        else:
            raise ValueError(f'Invalid directory name: {dir_name}')

    return split_files_by_label(file_paths, get_label_from_path, train_size, test_size, random_seed)
