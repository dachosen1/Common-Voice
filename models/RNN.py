import torch
import torch.nn as nn


class SingleRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons)  # 4 X 1
        self.Wy = torch.randn(n_neurons, n_neurons)  # 1 X 1
        self.b = torch.zeros(1, n_neurons)  # 1 X 4

    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b)  # 4 X 1
        self.Y1 = torch.tanh(
            torch.mm(self.Y0, self.Wy) + torch.mm(X1, self.Wx) + self.b
        )  # 4 X 1

        return self.Y0, self.Y1
