from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Union, Self, Optional, Callable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.selector import FilterSelectorCombinations

class ModelAdapterInterface(ABC):
    @abstractmethod
    def load(self, path: str = None, inline: Optional[bool] = True):
        pass

    @abstractmethod
    def get_test_data(self, **kwargs):
        pass

    @abstractmethod
    def get_criterion(self, **kwargs):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self, mode=True):
        pass

    @abstractmethod
    def evaluate(self,
                 batch_size: int = None,
                 max_batches: int = None,
                 top: Union[List[int], int] = 1,
                 include_loss: bool =True,
                 show_top_k: int = 2
                 ):
        pass

    @abstractmethod
    def get_bias(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        pass

    @abstractmethod
    def get_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        pass

    @abstractmethod
    def get_train_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        pass

    @abstractmethod
    def set_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]], weights: torch.Tensor):
        pass

    @abstractmethod
    def inject_weights(self, weights_vector: List[torch.Tensor], injection_combinations: FilterSelectorCombinations, inline=False):
        pass