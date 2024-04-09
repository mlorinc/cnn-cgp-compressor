from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Callable, Self
from pathlib import Path
import copy

class BaseModel(nn.Module):
    def __init__(self, model_path: str = None, model: nn.Module = None):
        super(BaseModel, self).__init__()
        self.model_path = model_path

    def _prepare(self):
        pass

    def _convert(self):
        pass

    def load(self, model_path: str, quantized: bool = False):
        model_state_dict: nn.Module = torch.load(model_path)

        if quantized:
            self.eval()
            self._prepare()
            self._convert()

        self.load_state_dict(model_state_dict)
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
        clone = copy.deepcopy(self)
        clone.load_state_dict(self.state_dict())
        return clone

    def quantize(self, new_path: str = None):
        torch.quantization.quantize_dynamic(self, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8, mapping=None, inplace=True)

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

            if batch_count is not None and i % batch_count == batch_count - 1:
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
        best_validation_loss = float("inf")
        best_train_loss = float("inf")

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
            average_train_loss = running_loss / batch_count
            print(f"\ttrain loss: {average_train_loss:.12f}, validation Loss: {average_validation_loss:.12f}")

            best_train_loss = min(average_train_loss, best_train_loss)
            if average_validation_loss < best_validation_loss:
                patience_counter = 0
                best_validation_loss = average_validation_loss
                if on_improve:
                    on_improve(average_validation_loss)
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping: No improvement for {patience} epochs.")
                break

        print(f"Finished training of {self.__class__.__name__}")
        return best_train_loss, best_validation_loss

    def _evaluate(self,
                 dataset: Dataset,
                 criterion: nn.modules.loss._WeightedLoss,
                 batch_size: int = 32,
                 max_batches: int = None,
                 top: int = 1
                 ):
        original_train_mode = self.training
        try:
            self.eval()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            running_loss = 0
            total_samples = 0
            running_correct = 0
            with torch.inference_mode():
                for batch_index, (x, y) in enumerate(loader):
                    y_hat = self(x)
                    loss = criterion(y_hat, y)
                    running_loss += loss.item() * y.size(0)

                    _, predicted = y_hat.topk(top, dim=1)
                    correct = predicted.eq(y.view(-1, 1).expand_as(predicted))
                    running_correct += correct.sum().item()
                    total_samples += y.size(0)

                    if batch_index % 100 == 0:
                        print(f"batch {batch_index} acc: {100 * correct / y.size(0):.12f}%, loss: {loss / y.size(0):.12f}")
                    if max_batches is not None and batch_index >= max_batches:
                        break

            acc = 100 * running_correct / total_samples
            average_loss = running_loss / total_samples
            return acc, average_loss
        except Exception as e:
            raise e
        finally:
            self.train(mode=original_train_mode)

    def fit(self, batch_size: int = 32) -> Tuple[float, float]:
        raise NotImplementedError()

    def evaluate(self, batch_size: int = 32, max_batches: int = None, top: int=1):
        raise NotImplementedError()
    
def init(model_path: Optional[str]) -> BaseModel:
    return BaseModel(model_path)
