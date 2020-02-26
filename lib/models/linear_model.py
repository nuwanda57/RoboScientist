import torch.nn as nn


class LinearModel(nn.Module):
    """
    Simple linear NN.
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
