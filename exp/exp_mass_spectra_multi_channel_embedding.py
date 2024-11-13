import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

from trainer import train, test
from datasets import MassSpectraDataset
from model.resnet_1d import build_resnet_1d
from model.densenet_1d import build_densenet_1d
from model.efficientnet_1d import build_efficientnet_1d
from prepare_datasets import (
    prepare_canine_sarcoma_dataset,
    prepare_microorganisms_dataset,
    prepare_nsclc_dataset,
    prepare_crlm_dataset,
    prepare_rcc_dataset
)
from utils.tools import EarlyStopping
from utils.dataset_split import split_dataset
from utils.data_loader import load_bin_mass_spec_data_from_pickle
from utils.data_normalization import tic_normalization


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def get_bin_dataset_path(exp_args):

    if exp_args['dataset'] in ['canine_sarcoma_posion', 'microorganisms']:
        bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

        if os.path.exists(bin_dataset_dir) is False:
            os.makedirs(bin_dataset_dir)

        saved_bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_train.pkl"
        saved_bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_test.pkl"
    elif exp_args['dataset'] == 'glioblastoma':
        bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'])

        if os.path.exists(bin_dataset_dir) is False:
            os.makedirs(bin_dataset_dir)

        saved_bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_train.pkl"
        saved_bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_test.pkl"
    elif exp_args['dataset'] in ['rcc_posion', 'nsclc', 'crlm', 'chd_urine', 'chd_serum']:
        bin_dataset_dir = os.path.join(exp_args['root_dir'], exp_args['dataset_dir'].replace('raw', f"bin/bin_{exp_args['bin_size']}"))

        if os.path.exists(bin_dataset_dir) is False:
            os.makedirs(bin_dataset_dir)

        saved_bin_train_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_train.pkl"
        saved_bin_test_dataset_path = f"{bin_dataset_dir}/{exp_args['dataset']}_classes_{exp_args['num_classes']}_bin_{exp_args['bin_size']}_rt_binning_window_{exp_args['rt_binning_window']}_test.pkl"
    else:
        raise ValueError(f'Unknown dataset: {exp_args["dataset"]}')

    return saved_bin_train_dataset_path, saved_bin_test_dataset_path


def prepare_dataset(exp_args, label_mapping):
    X_train, y_train = None, None
    X_test, y_test = None, None

    saved_bin_train_dataset_path, saved_bin_test_dataset_path = get_bin_dataset_path(exp_args)

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

        if exp_args['dataset'] == 'canine_sarcoma_posion':
            X_train, y_train, X_test, y_test = prepare_canine_sarcoma_dataset(
                exp_args=exp_args,
                label_mapping=label_mapping,
            )
        elif exp_args['dataset'] == 'microorganisms':
            X_train, y_train, X_test, y_test = prepare_microorganisms_dataset(
                exp_args=exp_args,
                label_mapping=label_mapping,
            )
        elif exp_args['dataset'] == 'nsclc':
            X_train, y_train, X_test, y_test = prepare_nsclc_dataset(
                exp_args=exp_args,
                label_mapping=label_mapping
            )
        elif exp_args['dataset'] == 'crlm':
            X_train, y_train, X_test, y_test = prepare_crlm_dataset(
                exp_args=exp_args,
                label_mapping=label_mapping
            )
        elif exp_args['dataset'] == 'rcc_posion':
            X_train, y_train, X_test, y_test = prepare_rcc_dataset(
                exp_args=exp_args,
                label_mapping=label_mapping
            )

        print(f'X_train.shape: {X_train.shape} y_train.shape: {y_train.shape}')
        print(f'X_test.shape: {X_test.shape} y_test.shape: {y_test.shape}')

        return X_train, y_train, X_test, y_test


def exp(exp_args, save_dir, label_mapping, device, use_multi_gpu=False):
    if 'Embedding' in exp_args['model_name']:
        print(f"{exp_args['model_name']} {exp_args['dataset']}_dataset num_classes {exp_args['num_classes']} in_channels: {exp_args['in_channels']} spectrum_dim: {exp_args['spectrum_dim']} embedding_channels: {exp_args['embedding_channels']} embedding_dim: {exp_args['embedding_dim']}")
        exp_dir_name = (f"{exp_args['model_name']}_{exp_args['dataset']}_dataset_num_classes_{exp_args['num_classes']}_in_channels_{exp_args['in_channels']}_spectrum_dim_{exp_args['spectrum_dim']}"
                        f"embedding_channels_{exp_args['embedding_channels']}_embedding_dim_{exp_args['embedding_dim']}_batch_size_{exp_args['batch_size']}")
        exp_dir = os.path.join(
            save_dir,
            exp_dir_name
        )

    else:
        print(f"{exp_args['model_name']} {exp_args['dataset']}_dataset num_classes {exp_args['num_classes']} in_channels: {exp_args['in_channels']} spectrum_dim: {exp_args['spectrum_dim']}")
        exp_dir_name = (f"{exp_args['model_name']}_{exp_args['dataset']}_dataset_num_classes_{exp_args['num_classes']}_in_channels_{exp_args['in_channels']}_"
                        f"spectrum_dim_{exp_args['spectrum_dim']}_batch_size_{exp_args['batch_size']}")
        exp_dir = os.path.join(
            save_dir,
            exp_dir_name
        )

    exp_name = f"{exp_args['model_name']}_train_{exp_args['dataset']}_dataset.pth"

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    model_summary = None
    all_metrics_results = []

    X_train, y_train, X_test, y_test = prepare_dataset(
        exp_args=exp_args,
        label_mapping=label_mapping,
    )

    if exp_args['is_normalization']:
        # Log transformation
        # print("Applying log transformation to X_train and X_test...")
        # X_train = log_transform(X_train)
        # X_test = log_transform(X_test)

        # TIC normalization
        print("Applying TIC normalization to X_train and X_test...")
        X_train = tic_normalization(X_train)
        X_test = tic_normalization(X_test)

        print("Normalization complete.")

    X_train, y_train, X_valid, y_valid = split_dataset(
        X_train,
        y_train,
        train_size=0.9,
        test_size=0.1,
    )

    train_loader = DataLoader(
        MassSpectraDataset(X_train, y_train),
        batch_size=exp_args['batch_size'],
        shuffle=True
    )

    valid_loader = DataLoader(
        MassSpectraDataset(X_valid, y_valid),
        batch_size=exp_args['batch_size'],
        shuffle=False
    )

    test_loader = DataLoader(
        MassSpectraDataset(X_test, y_test),
        batch_size=exp_args['batch_size'],
        shuffle=False
    )

    model = None

    if 'ResNet' in exp_args['model_name']:
        model = build_resnet_1d(exp_args)
    elif 'DenseNet' in exp_args['model_name']:
        model = build_densenet_1d(exp_args)
    elif 'EfficientNet' in exp_args['model_name']:
        model = build_efficientnet_1d(exp_args)

    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        device = torch.device("cuda:0")  # set the main GPU as cuda:0
        model = model.to(device)  # move the model to the main GPU
        model = nn.DataParallel(model)  # wrap the model with DataParallel for multi-GPU support 
    else:
        model = model.to(device)

    class_weights = compute_class_weight('balanced', classes=list(label_mapping.values()), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizers = [optimizer]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-12)
    schedulers = [scheduler]

    if exp_args['is_early_stopping']:
        early_stopping = EarlyStopping(patience=exp_args['patience'])
    else:
        early_stopping = None

    model_summary = summary(
        model,
        input_size=(
            exp_args['batch_size'],
            exp_args['spectrum_dim']
        )
    )

    # save model summary to txt
    with open(os.path.join(exp_dir, f"{exp_dir_name}_model_summary.txt"), 'w', encoding='utf-8') as file:
        file.write(str(model_summary))

    train(
        model,
        exp_dir,
        exp_name,
        train_loader,
        valid_loader,
        criterion,
        optimizers,
        schedulers,
        early_stopping,
        exp_args['epochs'],
        device,
        is_early_stopping=exp_args['is_early_stopping'],
        is_metrics_visualization=True
    )

    accuracy, precision, recall, f1_score = test(
        model,
        exp_dir,
        exp_name,
        test_loader,
        criterion,
        label_mapping,
        device,
        metric_args=None,
        is_metrics_visualization=True
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

    return exp_dir, exp_name, metrics_result


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra Embedding')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/embedding', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='AeEmbeddingLeNet', help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--file_extension', type=str, default='h5', help='File extension')
    # bin
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--spectrum_dim', type=int, default=15000, help='Spectrum dimension')
    parser.add_argument('--bin_size', type=float, default=0.1, help='Bin size')
    parser.add_argument('--rt_binning_window', type=int, default=10, help='Retention time binning window')
    # embedding
    parser.add_argument('--embedding_channels', type=int, default=256, help='Number of embedding channels')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')

    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--is_augmentation', action='store_true', help='Use augmentation')
    parser.add_argument('--is_normalization', action='store_true', help='Use normalization')
    parser.add_argument('--is_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--iteration', type=int, default=10, help='Number of iterations')

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.is_early_stopping:
        if args.patience is None:
            args.patience = 10

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_dirs = {
        'canine_sarcoma_posion': 'data/Canine_sarcoma/raw/positive',  # 100-1600 Da spectrum_dim 15000
        'microorganisms': 'data/Microorganisms/raw',  # 100-2000 Da spectrum_dim 19000
        'nsclc': 'data/NSCLC/raw',  # spectrum_dim 12000
        'crlm': 'data/CRLM/raw/mzML',  # spectrum_dim 12000
        'rcc_posion': 'data/RCC/raw/positive',  # spectrum_dim 9900
    }

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
        'microorganisms_3': {
            'Gram negative': 0,
            'Gram positive': 1,
            'Yeast': 2
        },
        'microorganisms_5': {
            'Staphylococcus aureus': 0,
            'E. coli D31': 1,
            'Pseudomonas aeruginosa': 2,
            'Enterococcus faecalis': 3,
            'Candida albicans': 4
        },
        'nsclc': {'ADC': 0, 'SCC': 1},
        'crlm': {'Control': 0, 'CRLM': 1},
        'rcc': {'Control': 0, 'RCC': 1}
    }

    label_mapping = None

    if 'canine_sarcoma' in args.dataset:
        label_mapping = label_mappings[f'canine_sarcoma_{args.num_classes}']
    elif 'microorganisms' in args.dataset:
        label_mapping = label_mappings[f'microorganisms_{args.num_classes}']
    elif 'nsclc' in args.dataset:
        label_mapping = label_mappings['nsclc']
    elif 'crlm' in args.dataset:
        label_mapping = label_mappings['crlm']
    elif 'rcc' in args.dataset:
        label_mapping = label_mappings['rcc']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    exp_args = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'root_dir': args.root_dir,
        'dataset_dir': dataset_dirs[args.dataset],

        'in_channels': args.in_channels,
        'spectrum_dim': args.spectrum_dim,
        'bin_size': args.bin_size,
        'rt_binning_window': args.rt_binning_window,
        'embedding_channels': args.embedding_channels,
        'embedding_dim': args.embedding_dim,

        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'is_augmentation': args.is_augmentation,
        'is_normalization': args.is_normalization,
        'is_early_stopping': args.is_early_stopping,
        'patience': args.patience,
        'random_seed': 3407,
    }

    exp_dir, trained_model_name, metrics_results = exp(
        exp_args=exp_args,
        save_dir=save_dir,
        label_mapping=label_mapping,
        device=device,
        use_multi_gpu=args.use_multi_gpu
    )

    # metrics_statistics = calculate_metrics_statistics(metrics_results)

    if 'Embedding' in exp_args['model_name']:
        print(f"\n{exp_args['model_name']} {args.dataset} in_channels: {exp_args['in_channels']}, spectrum_dim: {exp_args['spectrum_dim']}, "
              f"embedding_channels: {exp_args['embedding_channels']}, embedding_dim: {exp_args['embedding_dim']}, num_classes: {exp_args['num_classes']}, "
              f"batch_size: {exp_args['batch_size']}, epochs: {exp_args['epochs']}")

    else:
        print(f"\n{exp_args['model_name']} {args.dataset} in_channels: {exp_args['in_channels']}, spectrum_dim: {exp_args['spectrum_dim']}, num_classes: {exp_args['num_classes']}, "
              f"batch_size: {exp_args['batch_size']}, epochs: {exp_args['epochs']}")

    for metric, result in metrics_results.items():
        print(f'{metric}: {result:.4f}')


if __name__ == '__main__':
    main()











