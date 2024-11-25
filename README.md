# MSMCE: Classification Method for Mass Spectrometry Using Deep Multi-Channel Embedded Representation Integrated with Convolutional Neural Networks
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

[//]: # (This is the origin Pytorch implementation of MCE &#40;Multi Channel Embedding&#41; in the following paper: )
[//]: # ([Classification Method for Mass Spectrometry Using Deep Multi-Channel Embedded Representation Integrated with Convolutional Neural Networks]&#40;https://arxiv.org/abs/2012.07436&#41;.)

## Abstract
Mass spectrometry analysis plays a critical role in the biomedical field. 
However, the high dimensionality and complexity of mass spectrometry data present significant challenges for effective feature extraction and classification. 
Deep learning has become a dominant approach in data analysis, yet many existing methods for mass spectrometry data are limited in their feature representation capabilities. 
These methods often rely on single-channel representations, which are insufficient for capturing the deep and intricate information within mass spectrometry vectors. 
To address these limitations, we proposes a `deep learning-based multi-channel embedding representation method for mass spectrometry. 
This approach combines multilayer perceptron and convolutional neural network and performing embedding concatenation along the channel dimension to substantially enhance enhances the representation capability of mass spectrometry features. 
Experimental results on five public datasets demonstrate that the proposed method not only achieves significant improvements in classification performance but also reduces GPU memory usage and improves training efficiency, highlighting its effectiveness and generalizability in mass spectrometry data analysis.

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
or
```bash
conda env create -f environment.yml
```

## Data
The required data files should be put into `data/` folder. Note that the input of each dataset is tic normalized in this implementation.

The MS raw data can be downloaded from the source and the processed data can be downloaded here.
- [Google Drive](https://drive.google.com/drive/folders/16CIJCkPArsCuJrTgT7Y20jWUpHtflFNH?usp=sharing)

## Usage
Commands for training and testing the model with *MCE* on Dataset NSCLC, CRLM, RCC and Canine Sarcoma, Microorganisms respectively:

```bash
# NSCLC
python exp_mass_spectra_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset nsclc --num_classes 12 --in_channels 1 --spectrum_dim 15000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 128 --epochs 64 --patience 20 --is_normalization

# CRLM
python exp_mass_spectra_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset crlm --num_classes 12 --in_channels 1 --spectrum_dim 15000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 128 --epochs 64 --patience 20 --is_normalization
  
# RCC
python exp_mass_spectra_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset rcc_posion --num_classes 12 --in_channels 1 --spectrum_dim 15000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 128 --epochs 64 --patience 20 --is_normalization 

# Canine Sarcoma
python exp_mass_spectra_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset canine_sarcoma_posion --num_classes 12 --in_channels 1 --spectrum_dim 15000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 32 --epochs 64 --patience 20 --is_normalization 

# Microorganisms
python exp_mass_spectra_multi_channel_embedding.py --model_name MultiChannelEmbeddingResNet50 --dataset microorganisms --num_classes 5 --in_channels 1 --spectrum_dim 19000 --embedding_channels 256 --embedding_dim 1024 --device cuda:0 --batch_size 8 --epochs 64 --patience 20 --is_normalization 
```

More parameter information please refer to `exp_mass_spectra_multi_channel_embedding.py`.

We provide a more detailed and complete command description for training and testing the model:

```python
python main.py --root_dir <root_dir> --save_dir <save_dir> --model_name <model> --dataset <dataset> \
--in_channels <in_channels> --spectrum_dim <spectrum_dim> --bin_size <bin_size> --rt_binning_window <rt_binning_window> \
--embedding_channels <embedding_channels> --embedding_dim <embedding_dim> --num_classes <num_classes> --batch_size <batch_size> \
--epochs <epochs> --device <cuda:x> --use_multi_gpu --is_augmentation --is_normalization --is_early_stopping --patience <patience>\ 
```

The detailed descriptions about the arguments are as following:

| Parameter name     | Description of parameter                                                                                                                                                                  |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| root_dir           | Root directory of the project (default: `../`).                                                                                                                                           |
| save_dir           | Directory to save model checkpoints.                                                                                                                                                      |
| model_name         | Name of the model to use. This can be set to `MultiChannelEmbeddingResNet18`, `MultiChannelEmbeddingResNet50`, `MultiChannelEmbeddingDenseNet121`, `MultiChannelEmbeddingEfficientNetB0`. |
| dataset            | Name of the dataset to use (required).                                                                                                                                                    |
| in_channels        | Number of input channels for the model (default: `1`).                                                                                                                                    |
| spectrum_dim       | Spectrum dimension of the input data.                                                                                                                                                     |
| bin_size           | Size of bins for spectrum data (default: `0.1`).                                                                                                                                          |
| rt_binning_window  | Retention time binning window size (default: `10`).                                                                                                                                       |
| embedding_channels | Number of embedding channels (default: `256`).                                                                                                                                            |
| embedding_dim      | Dimension of the embedding vector (default: `256`).                                                                                                                                       |
| num_classes        | Number of classes in the dataset.                                                                                                                                                         |
| batch_size         | Size of each training batch.                                                                                                                                                              |
| epochs             | Number of training epochs (default: `64`).                                                                                                                                                |
| device             | Device to use for training (e.g., cpu, cuda, default: None, which auto-selects based on hardware).                                                                                        |
| use_multi_gpu      | Flag to enable training on multiple GPUs.                                                                                                                                                 |
| is_augmentation    | Flag to enable data augmentation during training.                                                                                                                                         |
| is_normalization   | Flag to enable normalization of the input data.                                                                                                                                           |
| is_early_stopping  | Flag to enable early stopping based on validation performance.                                                                                                                            |
| patience           | Number of epochs to wait before early stopping (default: `20`).                                                                                                                           |
