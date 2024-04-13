from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Callable, Self
from pathlib import Path
import copy
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseModel(ABC, nn.Module):
    def __init__(self, model_path: str = None):
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @abstractmethod
    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_validation_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_criterion(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_optimizer(self, **kwargs) -> optim.Optimizer:
        raise NotImplementedError()

    def load(self, model_path: Optional[str] = None):
        assert model_path or self.model_path is not None
        state = torch.load(model_path or self.model_path, map_location="cude:0" if self.device == "cuda" else None)
        self.load_state(state)
        self.to(self.device)
        return self

    def load_state(self, state):
        self.load_state_dict(state)
        return self        

    def get_state(self):
        return self.state_dict()

    def save(self, path: Optional[str] = None, inline=True, jit=False) -> Self:
        path = Path(path) if path is not None else Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not jit:
            torch.save(self.get_state(), path)
        else:
            jit_model = torch.jit.script(self)
            jit_model.save(path)
            return jit_model

        if inline:
            self.model_path = self.model_path or path
            return self
        else:
            model = self.clone();
            model.model_path = self.model_path or path
            return model

    def clone(self) -> Self:
        # Create a new instance of the same type
        clone = copy.deepcopy(self)
        clone.load_state(self.get_state())
        return clone

    def _fit_one_epoch(self, train_loader: DataLoader):
        running_loss = 0.0
        total_samples = 0
        optimizer = self.get_optimizer()
        with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader), leave=True) as batch_bar:
            criterion = self.get_criterion()
            for _, data in batch_bar:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_samples += labels.size(0)
                batch_bar.set_description(f"Loss: {running_loss / total_samples:.6f}")
        return running_loss / total_samples

    def fit(self,
              dataset: Dataset = None,
              batch_size: int = 32,
              shuffle: bool = True,
              validation_dataset_ratio: float = 0.2,
              epochs: int = 15,
              patience: int= 3,
              yield_on_improve=False):
        dataset = dataset or self.get_train_data()
        dataset_size = len(dataset)
        validation_set_size = int(validation_dataset_ratio * dataset_size)
        training_set_size = dataset_size - validation_set_size
        train_set, validation_set = random_split(dataset, [training_set_size, validation_set_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        # Number of epochs with no improvement before stopping
        patience_counter = 0
        best_validation_loss = float("inf")
        best_train_loss = float("inf")

        criterion = self.get_criterion()
        optimizer = self.get_optimizer()
        with tqdm(range(epochs), unit="Epoch", total=epochs, leave=True) as epoch_bar:
            # Fine-tune the model for a few epochs
            for epoch in epoch_bar:
                self.train(True)
                train_average_loss = self._fit_one_epoch(train_loader)
                
                # Disable dropout and use population
                # statistics for batch normalization.
                self.eval()
                validation_loss = 0.0
                with torch.no_grad():
                    for i, data in enumerate(validation_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        loss = criterion(self(inputs), labels)
                        validation_loss += loss.item()

                average_validation_loss = validation_loss / (i + 1)
                epoch_bar.set_description(f"Train Loss: {train_average_loss:.6f}, Validation Loss: {average_validation_loss:.6f}")

                best_train_loss = min(train_average_loss, best_train_loss)
                if average_validation_loss < best_validation_loss:
                    patience_counter = 0
                    best_validation_loss = average_validation_loss
                    if yield_on_improve:
                        yield best_train_loss, average_validation_loss
                else:
                    patience_counter += 1

                # Early stopping condition
                if patience_counter >= patience:
                    print(f"Early stopping: No improvement for {patience} epochs.")
                    break

        return best_train_loss, best_validation_loss

    def get_split_train_validation_loaders(self, 
                                    validation_dataset_ratio: float = 0.2,
                                    batch_size: int = 32,
                                    shuffle: bool = True,):
        dataset = self.get_train_data()
        dataset_size = len(dataset)
        validation_set_size = int(validation_dataset_ratio * dataset_size)
        training_set_size = dataset_size - validation_set_size
        train_set, validation_set = random_split(dataset, [training_set_size, validation_set_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader    