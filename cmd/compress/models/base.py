import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Callable, Self
from pathlib import Path

class BaseModel(nn.Module):
    def __init__(self, model_path: str = None):
        super(BaseModel, self).__init__()
        self.model_path = model_path

    def load(self, model_path: str):
        model: nn.Module = torch.load(model_path)

        if isinstance(model, nn.Module):
            return model
        else:
            self.load_state_dict(model)
            return self

    def save(self, path: str, save_model: bool = False):
        path = Path(path) if path is not None else Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not save_model:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, path)
        self.model_path = self.model_path or path

    def clone(self) -> Self:
        # Create a new instance of the same type
        clone = type(self)()
        clone.load_state_dict(self.state_dict())
        return clone

    def quantize(self):
        torch.quantization.quantize_dynamic(self, dtype=torch.qint8, mapping=None, inplace=True)
    
    def _fit_one_epoch(self, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.modules.loss._WeightedLoss, batch_count: int = 1000):
        running_loss = 0.0
        last_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % batch_count == batch_count - 1:
                # loss per batch
                last_loss = running_loss / batch_count
                print(f"\t Batch {i+1} loss: {last_loss}")
                running_loss = 0.
        return last_loss

    def _fit(self,
              dataset: Dataset,
              optimizer: optim.Optimizer,
              criterion: nn.modules.loss._WeightedLoss,
              on_improve: Optional[Callable] = None,
              batch_size: int = 32,
              shuffle: bool = True,
              validation_dataset_ratio: float = 0.2,
              epochs: int = 15,
              patience: int=3,
              batch_count: int = 1000):
        dataset_size = len(dataset)
        validation_set_size = int(validation_dataset_ratio * dataset_size)
        training_set_size = dataset_size - validation_set_size
        train_set, validation_set = random_split(dataset, [training_set_size, validation_set_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        # Number of epochs with no improvement before stopping
        patience_counter = 0
        best_loss = float("inf")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Fine-tune the model for a few epochs
        for epoch in range(epochs):
            print(f"epoch {epoch + 1}:")
            self.train(True)
            running_loss = self._fit_one_epoch(device, train_loader, optimizer, criterion, batch_count=batch_count)
            
            # Disable dropout and use population
            # statistics for batch normalization.
            self.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    loss = criterion(self(inputs), labels)
                    validation_loss += loss.item()

            average_validation_loss = validation_loss / (i + 1)
            print(f"\ttrain loss: {(running_loss / batch_count):.12f}, validation Loss: {average_validation_loss:.12f}")

            if average_validation_loss < best_loss:
                patience_counter = 0
                best_loss = average_validation_loss
                if on_improve:
                    on_improve(average_validation_loss)
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping: No improvement for {patience} epochs.")
                break

        print(f"Finished training of {self.__class__.__name__}")

    def _evaluate(self,
                 dataset: Dataset,
                 criterion: nn.modules.loss._WeightedLoss,
                 batch_size: int = 32
                 ):
        original_train_mode = self.training
        try:
            self.eval()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                running_loss = 0
                total = 0
                correct = 0
                for data in loader:
                    x, y = data
                    y_hat = self(x)
                    loss = criterion(y_hat, y)
                    running_loss = loss.item()
                    _, predicted = torch.max(y_hat.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            acc = 100 * correct / total
            test_loss = running_loss / len(loader)
            return acc, test_loss
        finally:
            self.train(mode=original_train_mode)

    def fit(self, batch_size: int = 32):
        raise NotImplementedError()

    def evaluate(self, batch_size: int = 32):
        raise NotImplementedError()
    
def init(model_path: Optional[str]) -> BaseModel:
    return BaseModel(model_path)
