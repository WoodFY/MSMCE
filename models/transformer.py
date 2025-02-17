import torch
import torch.nn as nn

from models.embedding.learnable_embedding import MultiChannelEmbedding


class Transformer(nn.Module):

    def __init__(self, spectrum_dim, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.input_dim = input_dim
        # spectrum split into tokens
        self.seq_len = spectrum_dim // input_dim
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
        new_length = (x.size(-1) // self.input_dim) * self.input_dim
        x = x[:, :new_length]
        # (batch_size, spectrum_dim (new_length)) -> (batch_size, seq_len, input_dim)
        x = x.view(x.size(0), self.seq_len, -1)
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.embedding_module(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        # (batch_size, seq_len + 1, hidden_dim)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer_encoder(x)
        # (batch_size, hidden_dim)
        cls_output = x[:, 0, :]
        out = self.classifier(self.dropout(cls_output))
        return out


class EmbeddingTransformer(Transformer):

    def __init__(self, spectrum_dim, input_dim, embedding_dim, num_heads, num_layers, num_classes, embedding_type=None, embedding_module=None):
        super().__init__(spectrum_dim=spectrum_dim, input_dim=input_dim, hidden_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
        self.embedding_type = embedding_type
        self.embedding_module = embedding_module

    def forward(self, x):
        if self.embedding_module:
            if self.embedding_type == 'MultiChannelEmbedding':
                x = self.embedding_module(x)
                cls_token = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = self.transformer_encoder(x)
                cls_output = x[:, 0, :]
                out = self.classifier(self.dropout(cls_output))
                return out
        else:
            raise ValueError("Invalid embedding type or module")


def build_transformer(args):
    input_dim = 1000
    hidden_dim = 1024
    num_heads = 4
    num_layers = 1

    model_name = args['model_name']

    if 'MultiChannelEmbedding' in model_name:
        embedding_type = 'MultiChannelEmbedding'
        embedding_module = MultiChannelEmbedding(
            spectrum_dim=args['spectrum_dim'],
            embedding_channels=args['embedding_channels'],
            embedding_dim=args['embedding_dim']
        )
    elif 'Embedding' in model_name and not any(substring in model_name for substring in ['MultiChannel']):
        raise ValueError(f"Invalid model name: {model_name}")
    else:
        embedding_type = None
        embedding_module = None

    if embedding_type:
        model_name = model_name.replace(embedding_type, '')

    # Transformer -> transformer
    base_model_name = model_name.lower()

    if embedding_type and embedding_module:
        return EmbeddingTransformer(
            spectrum_dim=args['spectrum_dim'],
            input_dim=input_dim,
            embedding_dim=args['embedding_dim'],
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=args['num_classes'],
            embedding_type=embedding_type,
            embedding_module=embedding_module
        )
    else:
        return Transformer(
            spectrum_dim=args['spectrum_dim'],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=args['num_classes']
        )


if __name__ == '__main__':
    model_args = {}
    model_args.update({
        # 'model_name': 'MultiChannelEmbeddingTransformer',
        'model_name': 'Transformer',
        'in_channels': 1,
        'spectrum_dim': 15000,
        'embedding_channels': 256,
        'embedding_dim': 1024,
        'num_classes': 3
    })
    model = build_transformer(model_args)

    # print models structure
    print(model)

    x = torch.randn(1, model_args['spectrum_dim'])

    output = model(x)
    print(output)
