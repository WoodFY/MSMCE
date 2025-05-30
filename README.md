# MSMCE: A Novel Representation Module for Classification of Raw Mass Spectrometry Data
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

## Abstract
Mass spectrometry (MS) analysis plays a crucial role in the biomedical field; however, the high dimensionality and complexity of MS data pose significant challenges for feature extraction and classification. Deep learning has become a dominant approach in data analysis, and while some deep learning methods have achieved progress in MS classification, their feature representation capabilities remain limited. Most existing methods rely on single-channel representations, which struggle to effectively capture structural information within MS data. To address these limitations, we propose a Multi-Channel Embedding Representation Module (MSMCE), which focuses on modeling inter-channel dependencies to generate multi-channel representations of raw MS data. Additionally, we implement a feature fusion mechanism by concatenating the initial encoded representation with the multi-channel embeddings along the channel dimension, significantly enhancing the classification performance of subsequent models. Experimental results on four public datasets demonstrate that the proposed MSMCE module not only achieves substantial improvements in classification performance but also enhances computational efficiency and training stability, highlighting its effectiveness in raw MS data classification and its potential for robust application across diverse datasets.

## Requirements

- Python 3.8
- pytorch == 2.3.1
- scikit_learn == 1.3.0
- numpy == 1.24.3
- pandas == 2.0.3
- matplotlib == 3.7.2


Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data
The required data files should be put into `datasets/` folder. Note that the input of each dataset is tic normalized in this implementation.

The MS raw data can be downloaded from the source and the processed data can be downloaded here.
- [Google Drive](https://drive.google.com/drive/folders/16CIJCkPArsCuJrTgT7Y20jWUpHtflFNH?usp=sharing)

## Usage
Commands for training and testing the model with *MSMCE* on Dataset NSCLC, CRLM, RCC and Canine Sarcoma respectively:

```bash
# NSCLC
python exp_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset nsclc --num_classes 2 --k_folds 6 --in_channels 1 --spectrum_dim 12000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 64 --epochs 64 --use_normalization

# CRLM
python exp_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset crlm --num_classes 2 --k_folds 6 --in_channels 1 --spectrum_dim 12000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 64 --epochs 64 --use_normalization
  
# RCC
python exp_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset rcc_posion --num_classes 2 --k_folds 6 --in_channels 1 --spectrum_dim 9900 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 64 --epochs 64 --use_normalization 

# Canine Sarcoma
python exp_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset canine_sarcoma_posion --num_classes 12 --k_folds 6 --in_channels 1 --spectrum_dim 15000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 32 --epochs 64 --use_normalization 

```

More parameter information please refer to `exp_multi_channel_embedding.py`.

We provide a more detailed and complete command description for training and testing the model:

```python
python exp_mass_spectra_multi_channel_embedding.py --root_dir <root_dir> --save_dir <save_dir> --model_name <model> --dataset <dataset> \
--in_channels <in_channels> --spectrum_dim <spectrum_dim> --bin_size <bin_size> --rt_binning_window <rt_binning_window> \
--embedding_channels <embedding_channels> --embedding_dim <embedding_dim> --num_classes <num_classes> --batch_size <batch_size> \
--epochs <epochs> --device <cuda:x> --use_multi_gpu --use_normalization --use_augmentation --use_early_stopping --patience <patience>\ 
```

The detailed descriptions about the arguments are as following:

| Parameter name     | Description of parameter                                                                                                                   |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| root_dir           | Root directory of the project (default: `../`).                                                                                            |
| save_dir           | Directory to save model checkpoints.                                                                                                       |
| model_name         | Name of the model to use. This can be set to `ResNet50`, `DenseNet121`, `EfficientNetB0`, `LSTM`, `Transformer`, `MultiChannelEmbedding~`. |
| dataset            | Name of the dataset to use (required).                                                                                                     |
| in_channels        | Number of input channels for the model (default: `1`).                                                                                     |
| spectrum_dim       | Spectrum dimension of the input data.                                                                                                      |
| bin_size           | Size of bins for spectrum data (default: `0.1`).                                                                                           |
| rt_binning_window  | Retention time binning window size (default: `10`second).                                                                                  |
| embedding_channels | Number of embedding channels (default: `256`).                                                                                             |
| embedding_dim      | Dimension of the embedding vector (default: `256`).                                                                                        |
| num_classes        | Number of classes in the dataset.                                                                                                          |
| k_folds            | Number of folds for cross-validation (default: `6`).                                                                                       |
| batch_size         | Size of each training batch.                                                                                                               |
| epochs             | Number of training epochs (default: `64`).                                                                                                 |
| device             | Device to use for training (e.g., cpu, cuda, default: None, which auto-selects based on hardware).                                         |
| num_workers        | Number of workers for data loading (default: `8`).                                                                                         |
| use_multi_gpu      | Flag to enable training on multiple GPUs.                                                                                                  |
| use_augmentation   | Flag to enable data augmentation during training.                                                                                          |
| use_normalization  | Flag to enable normalization of the input data.                                                                                            |
| use_early_stopping | Flag to enable early stopping based on validation performance.                                                                             |
| patience           | Number of epochs to wait before early stopping (default: `10`).                                                                            |
| random_seed        | Random seed for reproducibility (default: `3407`).                                                                                         |
