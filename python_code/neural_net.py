import torch.nn as nn
import torch

class NetModel(nn.Module):
    def __init__(self):
        super(NetModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward_prop(self, x):
        x = x.view(-1, )