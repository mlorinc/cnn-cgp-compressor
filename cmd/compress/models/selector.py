import torch
import torch.nn as nn
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from models.adapters.model_adapter_interface import ModelAdapterInterface

class FilterSelector(object):
    def __init__(self, selector: Union[str, Callable[[ModelAdapterInterface], nn.Conv2d]], inp: List, out: List, max_input_size=None, max_output_size=None) -> None:
        self.selector = selector
        self.inp = inp
        self.out = out
        self.max_input_size = max_input_size
        self.max_output_size = max_output_size
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

    @classmethod
    def all_weights(clf, layer: str):
        combinations = clf()
        combination = FilterSelectorCombination()
        combination.add(FilterSelector(layer, [(slice(None), slice(None), slice(None), slice(None))], [(slice(None), slice(None), slice(None), slice(None))]))
        combinations.add(combination)
        return combinations

class ConstantSelector(ABC):
    @abstractmethod
    def get_values(self) -> List:
        raise NotImplementedError()
    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError()

class ValuesSelector(ConstantSelector):
    def __init__(self, values: List[int]) -> None:
        self.values = values
    def get_values(self) -> List:
        return self.values
    def get_size(self) -> int:
        return len(self.values)

class ZeroSelector(ValuesSelector):
    def __init__(self, size: int) -> None:
        super().__init__([0] * size)
        
class ByteSelector(ValuesSelector):
    def __init__(self, size: int, zero=False) -> None:
        super().__init__([-128 / 2**i for i in range(0, size if not zero else size - 1)] + ([0] if zero else []) )
        self._zero  = zero

