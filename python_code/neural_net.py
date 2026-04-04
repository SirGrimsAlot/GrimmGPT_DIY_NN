import torch.nn as nn
import torch.nn.functional as F

class NetModel(nn.Module):
    def __init__(self, num_classes):
        super(NetModel, self).__init__()

        #Setup layers
        self.convolution1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)

        self.convolution2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.convolution3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        #Make model fully connected
        self.fc1 = nn.Linear(128*28*28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward_prop(self, x):
        x = self.pool(F.relu(self.bnorm1(self.convolution1(x))))
        x = self.pool(F.relu(self.bnorm2(self.convolution2(x))))
        x = self.pool(F.relu(self.bnorm3(self.convolution3(x))))

        x = x.view(-1, 128*28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
