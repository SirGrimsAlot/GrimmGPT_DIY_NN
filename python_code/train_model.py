import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import dataset as DATASET


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = DATASET('./train_set/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch)