import torch
import torch.nn as nn


class ANNMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        layers = []
        layers.append(nn.Linear(input_size, hidden_dim1))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim1))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim1, hidden_dim2))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim2))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim2, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.network(x)
