import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
import pandas as pd

from thop import profile
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.train_utils import train, test
from utils.ml_train_utils import train_test_ml
from datasets.datasets import MassSpectraDataset
from models.resnet_1d import build_resnet_1d
from models.densenet_1d import build_densenet_1d
from models.efficientnet_1d import build_efficientnet_1d
from models.lstm import build_lstm
from models.transformer import build_transformer
from datasets.prepare_datasets import (
    load_bin_mass_spec_data_from_pickle,
    prepare_canine_sarcoma_dataset,
    prepare_nsclc_dataset,
    prepare_crlm_dataset,
    prepare_rcc_dataset,
)
from callbacks.early_stopping import EarlyStopping
from utils.dataset_split import split_dataset
from utils.data_normalization import tic_normalization


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def get_bin_dataset_path(args):

    if args.dataset_name in ['canine_sarcoma']:
        bin_dataset_dir = os.path.join(args.root_dir, args.dataset_dir.replace('raw', f"bin/bin_{args.bin_size}"))

        if not os.path.exists(bin_dataset_dir):
            os.makedirs(bin_dataset_dir)

        saved_bin_train_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_train.pkl"
        saved_bin_test_dataset_path = f"{bin_dataset_dir}/{args.dataset_name}_classes_{args.num_classes}_bin_{args.bin_size}_test.pkl"
    elif args.dataset_name in ['rcc', 'nsclc', 'crlm']:
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

        if args.dataset_name == 'canine_sarcoma':
            X_train, y_train, X_test, y_test = prepare_canine_sarcoma_dataset(args)
        elif args.dataset_name == 'nsclc':
            X_train, y_train, X_test, y_test = prepare_nsclc_dataset(args)
        elif args.dataset_name == 'crlm':
            X_train, y_train, X_test, y_test = prepare_crlm_dataset(args)
        elif args.dataset_name == 'rcc':
            X_train, y_train, X_test, y_test = prepare_rcc_dataset(args)
        else:
            raise ValueError(f'Unknown dataset: {args.dataset_name}')

        print(f'X_train.shape: {X_train.shape} y_train.shape: {y_train.shape}')
        print(f'X_test.shape: {X_test.shape} y_test.shape: {y_test.shape}')

        return X_train, y_train, X_test, y_test


def exp(args):

    X_train, y_train, X_test, y_test = prepare_dataset(args)

    if args.use_normalization:
        # TIC normalization
        print("Applying TIC normalization to X_train and X_test...")
        X_train = tic_normalization(X_train)
        X_test = tic_normalization(X_test)

        print("Normalization complete.")

    # Machine learning models
    if args.model_name in ['SVM', 'RandomForest', 'XGBoost']:
        print(f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}")
        exp_dir_name = (f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}")
        exp_dir = os.path.join(
            args.save_dir,
            exp_dir_name
        )
        exp_model_name = f"{args.model_name}_train_{args.dataset_name}_{args.num_classes}"

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        if args.model_name == 'SVM':
            model = SVC(kernel='rbf', probability=True, random_state=3407)
        elif args.model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=10, random_state=3407)
        elif args.model_name == 'XGBoost':
            model = XGBClassifier(n_estimators=10, random_state=3407)
        else:
            raise ValueError(f'Unknown model: {args.model_name}')

        accuracy, precision, recall, f1_score = train_test_ml(
            exp_dir=exp_dir,
            exp_model_name=exp_model_name,
            model=model,
            train_set=(X_train, y_train),
            test_set=(X_test, y_test),
            label_mapping=args.label_mapping,
            metrics_visualization=True
        )

        metrics_result = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }

        # Save metrics to CSV using pandas
        time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        csv_file = os.path.join(exp_dir, f"{exp_dir_name}_metrics_{time_stamp}.csv")
        df = pd.DataFrame([metrics_result])
        df.to_csv(csv_file, index=False)

        return exp_dir, exp_model_name, metrics_result
    # Deep learning models
    else:
        if 'Embedding' in args.model_name:
            exp_dir_name = (
                f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_"
                f"in_channels_{args.in_channels}_spectrum_dim_{args.spectrum_dim}_"
                f"embedding_channels_{args.embedding_channels}_embedding_dim_{args.embedding_dim}_batch_size_{args.batch_size}"
            )
            print(exp_dir_name)
            exp_dir = os.path.join(
                args.save_dir,
                exp_dir_name
            )
        else:
            exp_dir_name = (
                f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_"
                f"in_channels_{args.in_channels}_spectrum_dim_{args.spectrum_dim}_batch_size_{args.batch_size}")
            print(exp_dir_name)
            exp_dir = os.path.join(
                args.save_dir,
                exp_dir_name
            )

        exp_model_name = f"{args.model_name}_train_{args.dataset_name}_num_classes_{args.num_classes}"

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        X_train, y_train, X_valid, y_valid = split_dataset(
            X_train,
            y_train,
            train_size=0.9,
            test_size=0.1,
        )

        train_loader = DataLoader(
            MassSpectraDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = DataLoader(
            MassSpectraDataset(X_valid, y_valid),
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

        class_weights = compute_class_weight('balanced', classes=np.array(list(args.label_mapping.values())), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-32)

        if args.use_early_stopping:
            early_stopping = EarlyStopping(patience=args.patience)
        else:
            early_stopping = None

        model_summary = summary(
            model,
            input_size=(
                args.batch_size,
                args.spectrum_dim
            )
        )

        # save models summary to txt
        with open(os.path.join(exp_dir, f"{exp_dir_name}_model_summary.txt"), 'w', encoding='utf-8') as file:
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
            with open(os.path.join(exp_dir, f"{exp_dir_name}_model_flops_params.txt"), 'w', encoding='utf-8') as file:
                file.write(f"FLOPs: {flops_str}\n")
                file.write(f"Params: {params_str}\n")

        train(
            exp_dir=exp_dir,
            exp_model_name=exp_model_name,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizers=[optimizer],
            schedulers=[scheduler],
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=args.device,
            use_early_stopping=args.use_early_stopping,
            metrics_visualization=True
        )

        accuracy, precision, recall, f1_score = test(
            exp_dir=exp_dir,
            exp_model_name=exp_model_name,
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            label_mapping=args.label_mapping,
            device=args.device,
            metrics_visualization=True
        )

        metrics_result = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }

        # Save metrics to CSV using pandas
        time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        csv_file = os.path.join(exp_dir, f"{exp_dir_name}_metrics_{time_stamp}.csv")
        df = pd.DataFrame([metrics_result])
        df.to_csv(csv_file, index=False)

        return exp_dir, exp_model_name, metrics_result


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra Embedding')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/embedding', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='MultiChannelEmbeddingResNet50', help='Model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset to use')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    # bin
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--spectrum_dim', type=int, default=15000, help='Spectrum dimension')
    parser.add_argument('--bin_size', type=float, default=0.1, help='Bin size')
    parser.add_argument('--rt_binning_window', type=int, default=10, help='Retention time binning window')
    # embedding
    parser.add_argument('--embedding_channels', type=int, default=256, help='Number of embedding channels')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--preload', action='store_true', help='Preload dataset into memory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--use_augmentation', action='store_true', help='Use augmentation')
    parser.add_argument('--use_normalization', action='store_true', help='Use normalization')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_multi_gpu:
        args.device = torch.device("cuda:0")

    if args.use_early_stopping:
        if args.patience is None:
            args.patience = 10

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    dataset_dirs = {
        'canine_sarcoma': 'datasets/Canine_sarcoma/raw/positive',  # 100-1600 Da spectrum_dim 15000
        'nsclc': 'datasets/NSCLC/raw',  # spectrum_dim 12000
        'crlm': 'datasets/CRLM/raw',  # spectrum_dim 12000
        'rcc': 'datasets/RCC/raw/positive',  # spectrum_dim 9900
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

    exp_dir, trained_model_name, metrics_results = exp(args)

    # metrics_statistics = calculate_metrics_statistics(metrics_results)

    if args.model_name in ['SVM', 'RandomForest', 'XGBoost']:
        print(f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}")
    else:
        if 'Embedding' in args.model_name:
            print(
                f"{args.model_name}_{args.dataset_name}_in_channels_{args.in_channels}_spectrum_dim_{args.spectrum_dim}_"
                f"embedding_channels_{args.embedding_channels}_embedding_dim_{args.embedding_dim}_num_classes_{args.num_classes}_"
                f"batch_size_{args.batch_size}_epochs_{args.epochs}"
            )
        else:
            print(
                f"{args.model_name}_{args.dataset_name}_in_channels_{args.in_channels}_spectrum_dim_{args.spectrum_dim}_num_classes_{args.num_classes}_"
                f"batch_size_{args.batch_size}_epochs_{args.epochs}"
            )

    for metric, result in metrics_results.items():
        print(f'{metric}: {result:.4f}')


if __name__ == '__main__':
    main()
