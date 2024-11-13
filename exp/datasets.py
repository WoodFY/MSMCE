import torch
from torch.utils.data import Dataset


class MassSpectraDataset(Dataset):

    def __init__(self, spectra, labels):
        self.spectra = torch.from_numpy(spectra).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]


