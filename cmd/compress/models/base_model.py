# Copyright 2024 Marián Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# base_model.py: Class extending PyTorch models by additional functionality such as loading, saving, training, cloning, etc.

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
    """
    An abstract base class for neural network models.

    Attributes:
        model_path (str): The path to the model file.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    
    def __init__(self, model_path: str = None):
        """
        Initializes the BaseModel with an optional model path.

        Args:
            model_path (str, optional): The path to the model file.
        """        
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @abstractmethod
    def get_train_data(self, **kwargs):
        """
        Abstract method to get training data.

        Args:
            **kwargs: Additional arguments for getting training data.
        
        Raises:
            NotImplementedError: This is an abstract method.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_validation_data(self, **kwargs):
        """
        Abstract method to get validation data.

        Args:
            **kwargs: Additional arguments for getting validation data.
        
        Raises:
            NotImplementedError: This is an abstract method.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        """
        Abstract method to get test data.

        Args:
            **kwargs: Additional arguments for getting test data.
        
        Raises:
            NotImplementedError: This is an abstract method.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_criterion(self, **kwargs):
        """
        Abstract method to get the loss criterion.

        Args:
            **kwargs: Additional arguments for getting the loss criterion.
        
        Raises:
            NotImplementedError: This is an abstract method.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_optimizer(self, **kwargs) -> optim.Optimizer:
        """
        Abstract method to get the optimizer.

        Args:
            **kwargs: Additional arguments for getting the optimizer.
        
        Returns:
            optim.Optimizer: The optimizer.
        
        Raises:
            NotImplementedError: This is an abstract method.
        """        
        raise NotImplementedError()

    def load(self, model_path: Optional[str] = None):
        """
        Loads the model from the specified path.

        Args:
            model_path (Optional[str], optional): The path to the model file.
        
        Returns:
            Self: The loaded model.
        """        
        assert model_path or self.model_path is not None
        state = torch.load(model_path or self.model_path, map_location="cude:0" if self.device == "cuda" else None)
        self.load_state(state)
        self.to(self.device)
        return self

    def load_state(self, state):
        """
        Loads the model state.

        Args:
            state: The state of the model.
        
        Returns:
            Self: The model with the loaded state.
        """        
        self.load_state_dict(state)
        self.to(self.device)
        return self        

    def get_state(self):
        """
        Gets the model state.

        Returns:
            dict: The state of the model.
        """        
        return self.state_dict()

    def save(self, path: Optional[str] = None, inline=True, jit=False) -> Self:
        """
        Saves the model to the specified path.

        Args:
            path (Optional[str], optional): The path to save the model.
            inline (bool, optional): Whether to save the model inline.
            jit (bool, optional): Whether to save the model as a JIT script.
        
        Returns:
            Self: The saved model.
        """        
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

    @abstractmethod
    def _create_self(self, *args) -> Self:
        """
        Abstract method to create a new instance of the model.

        Args:
            *args: Arguments for creating a new instance of the model.
        
        Returns:
            Self: A new instance of the model.
        
        Raises:
            NotImplementedError: This is an abstract method.
        """        
        raise NotImplementedError()

    def clone(self) -> Self:
        """
        Clones the current model instance.

        Returns:
            Self: A cloned instance of the current model.
        """        
        clone = self._create_self()
        clone.load_state(self.get_state())
        return clone

    def _fit_one_epoch(self, train_loader: DataLoader):
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): The data loader for the training data.
        
        Returns:
            float: The average loss over the epoch.
        """        
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
        """
        Trains the model.

        Args:
            dataset (Dataset, optional): The dataset to train on.
            batch_size (int, optional): The batch size for training.
            shuffle (bool, optional): Whether to shuffle the training data.
            validation_dataset_ratio (float, optional): The ratio of validation data to training data.
            epochs (int, optional): The number of epochs to train for.
            patience (int, optional): The number of epochs to wait for improvement before stopping.
            yield_on_improve (bool, optional): Whether to yield the best losses on improvement.
        
        Returns:
            Tuple[float, float]: The best training loss and validation loss.
        """        
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
                                    shuffle: bool = True):
        """
        Splits the dataset into training and validation sets and returns their data loaders.

        Args:
            validation_dataset_ratio (float, optional): The ratio of validation data to training data.
            batch_size (int, optional): The batch size for training and validation.
            shuffle (bool, optional): Whether to shuffle the training data.

        Returns:
            Tuple[DataLoader, DataLoader]: The data loaders for training and validation sets.
        """        
        dataset = self.get_train_data()
        dataset_size = len(dataset)
        validation_set_size = int(validation_dataset_ratio * dataset_size)
        training_set_size = dataset_size - validation_set_size
        train_set, validation_set = random_split(dataset, [training_set_size, validation_set_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader    