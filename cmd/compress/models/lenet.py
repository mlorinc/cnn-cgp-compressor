from typing import Optional
from models.base import BaseModel
import torch.nn as nn
import torch
from torch.utils.data import random_split, DataLoader
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
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

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
        # Load MNIST dataset
        dataset = self._get_train_data()
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        train_loss, val_loss = self._fit(dataset=dataset, optimizer=optimizer, criterion=criterion, on_improve=self._on_improve, batch_count=500, batch_size=16, epochs=100, patience=10)
        return train_loss, val_loss

    def evaluate(self, batch_size: int = 16, max_batches = None):
        # Load MNIST dataset
        dataset = self._get_test_data()
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        return self._evaluate(dataset=dataset, criterion=criterion, batch_size=batch_size, max_batches=max_batches)

    def _get_train_data(self):
        return torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)

    def _get_test_data(self):
        return torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)

    def _get_train_validation_data(self,
              batch_size: int = 32,
              shuffle: bool = True,
              validation_dataset_ratio: float = 0.2):
        dataset = self._get_train_data()
        dataset_size = len(dataset)
        validation_set_size = int(validation_dataset_ratio * dataset_size)
        training_set_size = dataset_size - validation_set_size
        train_set, validation_set = random_split(dataset, [training_set_size, validation_set_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader

def init(model_path: Optional[str]) -> LeNet5:
    return LeNet5(model_path)
