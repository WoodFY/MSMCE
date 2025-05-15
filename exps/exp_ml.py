import os
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from datasets.prepare_datasets import prepare_dataset
from utils.data_normalization import tic_normalization
from utils.ml_train_utils import train_test_ml
from utils.metrics import calculate_bootstrap_ci


def set_seeds(random_seed):
    """
    Sets random seeds for reproducibility.
    """
    np.random.seed(random_seed)
    print(f"Seeds set to: {random_seed}")


def machine_learning_model_and_param_grid(model_name, random_seed):
    if model_name == 'SVM':
        model = SVC(probability=True, random_state=random_seed)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=random_seed)
        param_grid = {
            'n_estimators': [5, 10, 50],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=random_seed)
        param_grid = {
            'n_estimators': [5, 10, 50],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
        }
    else:
        raise ValueError(f"Unknown ML model: {model_name}")
    return model, param_grid


def run_experiment(args):
    set_seeds(args.random_seed)
    print(f"--- Starting Experiment ---")
    print(f"--- K-Fold Cross-Validation on X_train, each fold tested on X_test ---")

    X_train, y_train, X_test, y_test = prepare_dataset(args)
    print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
    print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

    if args.use_normalization:
        # TIC normalization
        print("Applying TIC normalization to X_train and X_test...")
        X_train = tic_normalization(X_train)
        X_test = tic_normalization(X_test)
        print("Normalization complete.")

    exp_dir_name = (f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}")
    exp_base_dir = os.path.join(args.save_dir, exp_dir_name)
    if not os.path.exists(exp_base_dir):
        os.makedirs(exp_base_dir)

    print(exp_dir_name)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold_test_metrics_list = []

    for fold_idx, (train_fold_indices, valid_fold_indices) in enumerate(skf.split(X_train, y_train)):
        print(f"{args.model_name} Fold {fold_idx + 1}/{args.k_folds}")
        exp_model_name = f"{args.model_name}_kfold_{fold_idx + 1}_trained_on_{args.dataset_name}_{args.num_classes}"

        X_train_fold, y_train_fold = X_train[train_fold_indices], y_train[train_fold_indices]
        # X_valid_fold and y_valid_fold are not strictly used for final metric reporting in this setup,
        # but could be used for internal model checks or if hyperparameter tuning was done per fold.
        # For simplicity now, we train on X_fold_train and test on X_test_orig.
        X_valid_fold, y_valid_fold = X_train[valid_fold_indices], y_train[valid_fold_indices]
        print(f'X_train_fold.shape: {X_train_fold.shape}, y_train_fold.shape: {y_train_fold.shape}')
        print(f'X_valid_fold.shape: {X_valid_fold.shape}, y_valid_fold.shape: {y_valid_fold.shape}')
        print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

        if args.model_name == 'SVM':
            model = SVC(kernel='rbf', probability=True, random_state=args.random_seed)
        elif args.model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=10, random_state=args.random_seed)
        elif args.model_name == 'XGBoost':
            model = XGBClassifier(n_estimators=10, random_state=args.random_seed)
        else:
            raise ValueError(f'Unknown model: {args.model_name}')

        accuracy, precision, recall, f1_score = train_test_ml(
            model=model,
            train_set=(X_train_fold, y_train_fold),
            test_set=(X_test, y_test),
            label_mapping=args.label_mapping,
            exp_base_dir=exp_base_dir,
            exp_model_name=exp_model_name,
            metrics_visualization=True
        )

        fold_test_metrics_list.append({
            'Fold': fold_idx + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

    fold_test_metrics_df = pd.DataFrame(fold_test_metrics_list)
    mean_metrics = fold_test_metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean()
    std_metrics = fold_test_metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].std()

    summary_stats_list = []
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        metric_values = fold_test_metrics_df[metric].values
        ci_lower, ci_upper = calculate_bootstrap_ci(metric_values, random_seed=args.random_seed)
        summary_stats_list.append({
            'Metric': metric,
            'Mean_Test_on_Holdout': mean_metrics.get(metric, np.nan),
            'Std_Test_on_Holdout': std_metrics.get(metric, np.nan),
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper
        })

    print(exp_dir_name)
    summary_stats_df = pd.DataFrame(summary_stats_list)
    print("Summary Statistics (Mean, Std, 95% Bootstrap CI from K-Fold models tested on hold-out):")
    print(summary_stats_df)

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    fold_test_metrics_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_kfold_tested_on_holdout_metrics_{time_stamp}.csv")
    summary_stats_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_kfold_tested_on_holdout_summary_stats_{time_stamp}.csv")
    fold_test_metrics_df.to_csv(fold_test_metrics_csv_path, index=False)
    summary_stats_df.to_csv(summary_stats_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='ML Models with K-Fold Cross-Validation on Mass Spectrometry Dataset')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/embedding', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='MultiChannelEmbeddingResNet50', help='Model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset to use')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    # bin
    parser.add_argument('--bin_size', type=float, default=0.1, help='Bin size')
    parser.add_argument('--rt_binning_window', type=int, default=10, help='Retention time binning window (Unit: Seconds)')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--spectrum_dim', type=int, default=15000, help='Spectrum dimension')

    parser.add_argument('--use_normalization', action='store_true', help='Use normalization')
    parser.add_argument('--k_folds', type=int, default=6, help='Number of folds for K-Fold cross validation.')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    dataset_dirs = {
        'canine_sarcoma_posion': 'datasets/Canine_sarcoma/raw/positive',  # 100-1600 Da spectrum_dim 15000
        'nsclc': 'datasets/NSCLC/raw',  # 400-1600 Da spectrum_dim 12000
        'crlm': 'datasets/CRLM/raw',  # 400-1600 Da spectrum_dim 12000
        'rcc_posion': 'datasets/RCC/raw/positive',  # 70-1060 Da spectrum_dim 9900
    }
    args.dataset_dir = dataset_dirs[args.dataset_name]

    label_mappings = {
        'canine_sarcoma_2': {'Healthy': 0, 'Cancerous': 1},
        'canine_sarcoma_12': {
            'Healthy': 0,
            'Myxosarcoma': 1,
            'Fibrosarcoma': 2,
            'Hemangiopericytoma': 3,
            'Malignant peripheral nerve tumor': 4,
            'Osteosarcoma': 5,
            'Undifferentiated pleomorphic': 6,
            'Rhabdomyosarcoma': 7,
            'Splenic fibrohistiocytic nodules': 8,
            'Histiocytic sarcoma': 9,
            'Soft tissue sarcoma': 10,
            'Gastrointestinal stromal sarcoma': 11
        },
        'nsclc': {'ADC': 0, 'SCC': 1},
        'crlm': {'Control': 0, 'CRLM': 1},
        'rcc': {'Control': 0, 'RCC': 1},
    }

    if 'canine_sarcoma' in args.dataset_name:
        label_mapping = label_mappings[f"canine_sarcoma_{args.num_classes}"]
    elif 'nsclc' in args.dataset_name:
        label_mapping = label_mappings['nsclc']
    elif 'crlm' in args.dataset_name:
        label_mapping = label_mappings['crlm']
    elif 'rcc' in args.dataset_name:
        label_mapping = label_mappings['rcc']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset_name}')

    args.label_mapping = label_mapping

    run_experiment(args)


if __name__ == '__main__':
    main()
