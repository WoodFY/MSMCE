import torch
import torch.nn as nn

from models.embedding.learnable_embedding import MultiChannelEmbedding


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        elif num_layers == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid number of layers")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, spectrum_dim) -> (batch_size, spectrum_dim, 1)
        x = x.unsqueeze(-1)
        # lstm_out: (batch_size, spectrum_dim, hidden_size)
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
                x = self.embedding_module(x)
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                out = self.classifier(last_out)
                return out
        else:
            raise ValueError("Invalid embedding type or module")


def build_lstm(args):
    input_size = 1
    hidden_size = 32
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

    # LSTM -> lstm
    base_model_name = model_name.lower()

    if embedding_type and embedding_module:
        return EmbeddingLSTM(
            embedding_dim=args['embedding_dim'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=args['num_classes'],
            embedding_type=embedding_type,
            embedding_module=embedding_module
        )
    else:
        return LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=args['num_classes']
        )


if __name__ == '__main__':
    model_args = {}
    model_args.update({
        'model_name': 'MultiChannelEmbeddingLSTM',
        # 'model_name': 'LSTM',
        'in_channels': 1,
        'spectrum_dim': 15000,
        'embedding_channels': 256,
        'embedding_dim': 1024,
        'num_classes': 3
    })
    model = build_lstm(model_args)

    # print models structure
    print(model)

    x = torch.randn(1, model_args['spectrum_dim'])

    output = model(x)
    print(output)


