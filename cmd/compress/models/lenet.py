from typing import Optional, Self
from models.base_model import BaseModel
import torch.nn as nn
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class LeNet5(BaseModel):
    name = "lenet"
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    def __init__(self, model_path: str = None):
        super(LeNet5, self).__init__(model_path)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def _create_self(self, *args) -> Self:
        return LeNet5(self.model_path)

    def get_train_data(self):
        return torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=LeNet5.transforms)

    def get_test_data(self):
        return torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=LeNet5.transforms)

    def get_validation_data(self):
        raise ValueError("not supported")

    def get_criterion(self):
        return nn.CrossEntropyLoss()
    
    def get_optimizer(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def init(model_path: Optional[str]) -> LeNet5:
    return LeNet5(model_path)
