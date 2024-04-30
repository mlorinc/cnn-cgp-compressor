import torch
import torch.nn as nn
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from models.adapters.model_adapter import ModelAdapter

class FilterSelector(object):
    def __init__(self, selector: Union[str, Callable[[ModelAdapter], nn.Conv2d]], inp: List, out: List) -> None:
        self.selector = selector
        self.inp = inp
        self.out = out
    def clone(self):
        cloned = FilterSelector(self.selector, self.inp[:], self.out[:])
        return cloned

class FilterSelectorCombination(object):
    def __init__(self) -> None:
        self._selector: List[FilterSelector] = []
    def add(self, sel: FilterSelector):
        self._selector.append(sel)
    def get_selectors(self) -> List[FilterSelector]:
        return self._selector
    def clone(self):
        cloned = FilterSelectorCombination()
        cloned._selector = [sel.clone() for sel in self._selector]
        return cloned

class FilterSelectorCombinations(object):
    def __init__(self) -> None:
        self._combinations: List[FilterSelectorCombination] = []
    def add(self, combination: FilterSelectorCombination):
        self._combinations.append(combination)
    def get_combinations(self) -> List[FilterSelectorCombination]:
        return self._combinations
    def clone(self):
        cloned = FilterSelectorCombinations()
        cloned._combinations = [sel.clone() for sel in self._combinations]
        return cloned

class ConstantSelector(ABC):
    @abstractmethod
    def get_values(self) -> List:
        raise NotImplementedError()
    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError()

class ZeroSelector(ConstantSelector):
    def __init__(self, size: int) -> None:
        self.size = size
    def get_values(self) -> List:
        return [0] * self.size
    def get_size(self) -> int:
        return self.size
class ByteSelector(ConstantSelector):
    def __init__(self, size: int, zero=False) -> None:
        self.size = size
        self._zero  =zero
    def get_values(self) -> List:
        return [-128 / 2**i for i in range(0, self.size if not self._zero else self.size - 1)] + ([0] if self._zero else [])   
    def get_size(self) -> int:
        return self.size    
