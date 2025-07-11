import torch
import torch.nn as nn

import argparse

from models.embedding.multi_channel_embedding import MSMCE


class Transformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.input_dim = input_dim
        # linear projection to hidden dimension
        self.embedding_module = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.5,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.ndim == 2:
            batch_size = x.size(0)
            spectrum_dim = x.size(1)
            seq_len = spectrum_dim // self.input_dim
            x = x[:, :seq_len * self.input_dim]
            # (batch_size, spectrum_dim (new_length)) -> (batch_size, seq_len, input_dim)
            x = x.view(batch_size, seq_len, self.input_dim)
            # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
            x_embedded = self.embedding_module(x)
            cls_token = self.cls_token.expand(x_embedded.size(0), -1, -1)
            # (batch_size, seq_len + 1, hidden_dim)
            x = torch.cat([cls_token, x_embedded], dim=1)
        x = self.transformer_encoder(x)
        # (batch_size, hidden_dim)
        cls_output = x[:, 0, :]
        out = self.classifier(self.dropout(cls_output))
        return out


class EmbeddingTransformer(Transformer):

    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, num_classes, embedding_type=None, embedding_module=None):
        super().__init__(input_dim=input_dim, hidden_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
        self.embedding_type = embedding_type
        self.embedding_module = embedding_module

    def forward(self, x):
        if self.embedding_module:
            if self.embedding_type == 'MSMCE':
                x_embedded = self.embedding_module(x)
                cls_token = self.cls_token.expand(x_embedded.size(0), -1, -1)
                x_embedded = torch.cat([cls_token, x_embedded], dim=1)
                return super().forward(x_embedded)
        else:
            raise ValueError("Invalid embedding type or module")


def build_transformer(args):
    input_dim = 1000
    hidden_dim = 1024
    num_heads = 4
    num_layers = 1

    model_name = args.model_name

    if 'MSMCE' in model_name:
        embedding_type = 'MSMCE'
        embedding_module = MSMCE(
            spectrum_dim=args.spectrum_dim,
            embedding_channels=args.embedding_channels,
            embedding_dim=args.embedding_dim
        )
    else:
        embedding_type = None
        embedding_module = None

    if embedding_type:
        model_name = model_name.replace(embedding_type, '')

    # Transformer -> transformer
    base_model_name = model_name.lower()

    if embedding_type and embedding_module:
        return EmbeddingTransformer(
            input_dim=input_dim,
            embedding_dim=args.embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=args.num_classes,
            embedding_type=embedding_type,
            embedding_module=embedding_module
        )
    else:
        return Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=args.num_classes
        )


if __name__ == '__main__':
    args = {
        'model_name': 'MSMCE-Transformer',
        # 'model_name': 'Transformer',
        'in_channels': 1,
        'spectrum_dim': 15000,
        'embedding_channels': 256,
        'embedding_dim': 1024,
        'num_classes': 3
    }
    args = argparse.Namespace(**args)
    model = build_transformer(args)

    # print models structure
    print(model)

    x = torch.randn(1, args.spectrum_dim)

    output = model(x)
    print(output)
