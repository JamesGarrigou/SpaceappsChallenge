import torch.nn as nn
import torch.nn.functional as F


class STD(nn.Module):
    def __init__(self, n_features: int, hidden_units: int, distributional: bool):
        super(STD, self).__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units

        self.input_layer = nn.Linear(n_features, hidden_units)

        # predict value and variance
        self.hidden_layer = nn.Linear(hidden_units, 2 if distributional else 0)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        return self.hidden_layer(out)


def MLP_STD(n_features: int, hidden_units: int):
    return STD(n_features=n_features, hidden_units=hidden_units)
