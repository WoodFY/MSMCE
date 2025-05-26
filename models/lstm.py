import torch
import torch.nn as nn

import argparse

from models.embedding.learnable_embedding import MultiChannelEmbedding


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.input_size = input_size
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        elif num_layers == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid number of layers")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim == 2:
            batch_size = x.size(0)
            spectrum_dim = x.size(1)
            num_time_steps = spectrum_dim // self.input_size
            x = x[:, :num_time_steps * self.input_size]
            # x: (batch_size, num_time_steps, input_size)
            x = x.view(batch_size, num_time_steps, self.input_size)
        # lstm_out: (batch_size, num_time_steps, hidden_size)
        lstm_out, _ = self.lstm(x)
        # last_out: (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]
        out = self.classifier(last_out)
        return out


class EmbeddingLSTM(LSTM):

    def __init__(self, embedding_dim, hidden_size, num_layers, num_classes, embedding_type=None, embedding_module=None):
        super().__init__(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
        self.embedding_type = embedding_type
        self.embedding_module = embedding_module

    def forward(self, x):
        if self.embedding_module:
            if self.embedding_type == 'MultiChannelEmbedding':
                x_embedded = self.embedding_module(x)
                return super().forward(x_embedded)
        else:
            raise ValueError("Invalid embedding type or module")


def build_lstm(args):
    input_size = 1000
    hidden_size = 2048
    num_layers = 2

    model_name = args.model_name

    if 'MultiChannelEmbedding' in model_name:
        embedding_type = 'MultiChannelEmbedding'
        embedding_module = MultiChannelEmbedding(
            spectrum_dim=args.spectrum_dim,
            embedding_channels=args.embedding_channels,
            embedding_dim=args.embedding_dim
        )
    elif 'Embedding' in model_name and not any(substring in model_name for substring in ['MultiChannel']):
        raise ValueError(f"Invalid model name: {model_name}")
    else:
        embedding_type = None
        embedding_module = None

    if embedding_type:
        model_name = model_name.replace(embedding_type, '')

    # LSTM -> lstm
    base_model_name = model_name.lower()

    if embedding_type and embedding_module:
        return EmbeddingLSTM(
            embedding_dim=args.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=args.num_classes,
            embedding_type=embedding_type,
            embedding_module=embedding_module
        )
    else:
        return LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=args.num_classes
        )


if __name__ == '__main__':
    args = {
        'model_name': 'MultiChannelEmbeddingLSTM',
        # 'model_name': 'LSTM',
        'in_channels': 1,
        'spectrum_dim': 15000,
        'embedding_channels': 256,
        'embedding_dim': 1024,
        'num_classes': 3
    }
    args = argparse.Namespace(**args)
    model = build_lstm(args)

    # print models structure
    print(model)

    x = torch.randn(1, args.spectrum_dim)

    output = model(x)
    print(output)


