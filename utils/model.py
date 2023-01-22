import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        num_layers = 3
        layer_dim = 64
        self.layers = nn.ModuleList([
            nn.Linear(5, layer_dim),
            nn.ReLU(),
        ])
        for _ in range(num_layers - 1):
            self.layers.extend([
                nn.Linear(layer_dim, layer_dim),
                nn.ReLU(),
            ])
        self.out = nn.Linear(layer_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        x = torch.squeeze(x, -1)
        return x