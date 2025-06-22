import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader

import os
import time
import argparse
import numpy as np
import pandas as pd

from thop import profile
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from datasets.prepare_datasets import prepare_dataset
from datasets.datasets import MassSpectraDataset
from models.resnet_1d import build_resnet_1d
from models.densenet_1d import build_densenet_1d
from models.efficientnet_1d import build_efficientnet_1d
from models.lstm import build_lstm
from models.transformer import build_transformer
from callbacks.early_stopping import EarlyStopping
from utils.normalization import tic_normalization
from utils.train_utils import train, test
from utils.metrics import calculate_bootstrap_ci


def set_seeds(random_seed):
    """
    Sets random seeds for reproducibility.
    """
    np.random.seed(random_seed)
    print(f"Seeds set to: {random_seed}")


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

    if 'MSMCE' in args.model_name:
        exp_dir_name = (
            f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_"
            f"in_channels_{args.in_channels}_spectrum_dim_{args.spectrum_dim}_"
            f"embedding_channels_{args.embedding_channels}_embedding_dim_{args.embedding_dim}_batch_size_{args.batch_size}"
        )
    else:
        exp_dir_name = (
            f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_"
            f"in_channels_{args.in_channels}_spectrum_dim_{args.spectrum_dim}_batch_size_{args.batch_size}"
        )

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
        X_valid_fold, y_valid_fold = X_train[valid_fold_indices], y_train[valid_fold_indices]
        print(f'X_train_fold.shape: {X_train_fold.shape}, y_train_fold.shape: {y_train_fold.shape}')
        print(f'X_valid_fold.shape: {X_valid_fold.shape}, y_valid_fold.shape: {y_valid_fold.shape}')
        print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

        train_loader = DataLoader(
            MassSpectraDataset(X_train_fold, y_train_fold),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = DataLoader(
            MassSpectraDataset(X_valid_fold, y_valid_fold),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        test_loader = DataLoader(
            MassSpectraDataset(X_test, y_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        if 'ResNet' in args.model_name:
            model = build_resnet_1d(args)
        elif 'DenseNet' in args.model_name:
            model = build_densenet_1d(args)
        elif 'EfficientNet' in args.model_name:
            model = build_efficientnet_1d(args)
        elif 'LSTM' in args.model_name:
            model = build_lstm(args)
        elif 'Transformer' in args.model_name:
            model = build_transformer(args)
        else:
            raise ValueError(f'Unknown model: {args.model_name}')

        if args.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for training.')
            print("DataParallel typically expects model on primary GPU (cuda:0). Moving model to cuda:0 before DataParallel.")
            model = model.to(args.device)
            model = nn.DataParallel(model)  # Wrap the models with DataParallel for multi-GPU support
        else:
            model = model.to(args.device)

        class_weights = compute_class_weight('balanced', classes=np.array(list(args.label_mapping.values())), y=y_train_fold)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-32)

        if args.use_early_stopping:
            early_stopping = EarlyStopping(patience=args.patience)
        else:
            early_stopping = None

        model_summary = summary(model, input_size=(args.batch_size, args.spectrum_dim))
        summary_file_path = os.path.join(exp_base_dir, f"model_summary.txt")
        # save models summary to txt
        with open(summary_file_path, 'w', encoding='utf-8') as file:
            file.write(str(model_summary))

        if not args.use_multi_gpu:
            flops, params = profile(
                model,
                inputs=(torch.randn(1, args.spectrum_dim).to(args.device),)
            )
            flops_str = f'{flops / 1e9:.2f} GFLOPs'
            params_str = f'{params / 1e6:.2f}M Params'
            print(f"FLOPs: {flops_str}")
            print(f"Parameters: {params_str}")
            model_flops_params_file_path = os.path.join(exp_base_dir, f"model_flops_params.txt")
            with open(model_flops_params_file_path, 'w', encoding='utf-8') as file:
                file.write(f"FLOPs: {flops_str}\n")
                file.write(f"Params: {params_str}\n")
            # time.sleep(10000)

        train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=args.device,
            exp_base_dir=exp_base_dir,
            exp_model_name=exp_model_name,
            metrics_visualization=True
        )

        accuracy, precision, recall, f1_score = test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            label_mapping=args.label_mapping,
            device=args.device,
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
    fold_test_metrics_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_kfold_tested_on_holdout_metrics_{time_stamp}.csv")
    summary_stats_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_kfold_tested_on_holdout_summary_stats_{time_stamp}.csv")
    fold_test_metrics_df.to_csv(fold_test_metrics_csv_path, index=False)
    summary_stats_df.to_csv(summary_stats_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra Embedding')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/embedding', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='MultiChannelEmbeddingResNet50', help='Model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset to use')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    # bin
    parser.add_argument('--bin_size', type=float, default=0.1, help='Bin size')
    parser.add_argument('--rt_binning_window', type=int, default=10, help='Retention time binning window')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--spectrum_dim', type=int, default=15000, help='Spectrum dimension')
    # embedding
    parser.add_argument('--embedding_channels', type=int, default=256, help='Number of embedding channels')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')

    parser.add_argument('--k_folds', type=int, default=6, help='Number of folds for K-Fold cross validation.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--preload', action='store_true', help='Preload dataset into memory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--use_normalization', action='store_true', help='Use normalization')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_multi_gpu:
        args.device = torch.device("cuda:0")

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
