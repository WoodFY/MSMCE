import numpy as np

from sklearn.model_selection import train_test_split


def split_dataset(X, y, train_size=0.9, test_size=0.1, random_seed=3047):
    """
    Split the dataset into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def split_mass_spec_dataset(mz_arrays, intensity_arrays, labels, train_size=0.9, test_size=0.1, random_seed=3407):

    mz_train, mz_test, intensity_train, intensity_test, labels_train, labels_test = train_test_split(
        mz_arrays,
        intensity_arrays,
        labels,
        train_size=train_size,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )

    return mz_train, intensity_train, labels_train, \
        mz_test, intensity_test, labels_test
