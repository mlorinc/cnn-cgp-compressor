from typing import Optional
from models.base import BaseModel
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class LeNet5(BaseModel):
    name = "lenet"
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

    def _on_improve(self, average_validation_loss: float):
        self.save(self.model_path)
        print(f"Saved the best model with validation loss: {average_validation_loss:.6f}")

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

    def fit(self):
        # Define transformations and load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # Load MNIST dataset
        dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self._fit(dataset=dataset, optimizer=optimizer, criterion=criterion, on_improve=self._on_improve, batch_count=500, batch_size=16, epochs=100, patience=10)

    def evaluate(self, batch_size: int = 16):
        # Define transformations and load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # Load MNIST dataset
        dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        return self._evaluate(dataset=dataset, criterion=criterion, batch_size=batch_size)

def init(model_path: Optional[str]) -> LeNet5:
    return LeNet5(model_path)
