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
# selector.py: Various selector definitions used for choosing what weights to use when training or infering.

import torch
import torch.nn as nn
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from models.adapters.model_adapter_interface import ModelAdapterInterface

class FilterSelector(object):
    """
    Selector for filtering layers in a model.

    Args:
        selector (Union[str, Callable[[ModelAdapterInterface], nn.Conv2d]]): Selector function or name.
        inp (List): Input list.
        out (List): Output list.
        max_input_size (optional): Maximum input size. Defaults to None.
        max_output_size (optional): Maximum output size. Defaults to None.
    """    
    def __init__(self, selector: Union[str, Callable[[ModelAdapterInterface], nn.Conv2d]], inp: List, out: List, max_input_size=None, max_output_size=None) -> None:
        self.selector = selector
        self.inp = inp
        self.out = out
        self.max_input_size = max_input_size
        self.max_output_size = max_output_size
    def clone(self):
        """
        Clone the FilterSelector object.

        Returns:
            FilterSelector: Cloned object.
        """        
        cloned = FilterSelector(self.selector, self.inp[:], self.out[:])
        return cloned

class FilterSelectorCombination(object):
    """
    Combination of filter selectors.

    Attributes:
        _selector (List[FilterSelector]): List of filter selectors.
    """    
    def __init__(self) -> None:
        self._selector: List[FilterSelector] = []
    def add(self, sel: FilterSelector):
        """
        Add a filter selector to the combination.

        Args:
            sel (FilterSelector): Filter selector to add.
        """        
        self._selector.append(sel)
    def get_selectors(self) -> List[FilterSelector]:
        """
        Get the list of filter selectors.

        Returns:
            List[FilterSelector]: List of filter selectors.
        """        
        return self._selector
    def clone(self):
        """
        Clone the FilterSelectorCombination object.

        Returns:
            FilterSelectorCombination: Cloned object.
        """        
        cloned = FilterSelectorCombination()
        cloned._selector = [sel.clone() for sel in self._selector]
        return cloned

class FilterSelectorCombinations(object):
    """
    Container for multiple filter selector combinations.

    Attributes:
        _combinations (List[FilterSelectorCombination]): List of filter selector combinations.
    """    
    def __init__(self) -> None:
        self._combinations: List[FilterSelectorCombination] = []
    def add(self, combination: FilterSelectorCombination):
        """
        Add a filter selector combination.

        Args:
            combination (FilterSelectorCombination): Filter selector combination to add.
        """        
        self._combinations.append(combination)
    def get_combinations(self) -> List[FilterSelectorCombination]:
        """
        Get the list of filter selector combinations.

        Returns:
            List[FilterSelectorCombination]: List of filter selector combinations.
        """        
        return self._combinations
    def clone(self):
        """
        Clone the FilterSelectorCombinations object.

        Returns:
            FilterSelectorCombinations: Cloned object.
        """        
        cloned = FilterSelectorCombinations()
        cloned._combinations = [sel.clone() for sel in self._combinations]
        return cloned

    @classmethod
    def all_weights(clf, layer: str):
        """
        Create filter selector combinations for all weights.

        Args:
            layer (str): Layer name.

        Returns:
            FilterSelectorCombinations: Filter selector combinations for all weights.
        """        
        combinations = clf()
        combination = FilterSelectorCombination()
        combination.add(FilterSelector(layer, [(slice(None), slice(None), slice(None), slice(None))], [(slice(None), slice(None), slice(None), slice(None))]))
        combinations.add(combination)
        return combinations

class ConstantSelector(ABC):
    """
    Abstract base class for constant selectors.
    """    
    @abstractmethod
    def get_values(self) -> List:
        """
        Get the values of the selector.

        Returns:
            List: List of values.
        """        
        raise NotImplementedError()
    @abstractmethod
    def get_size(self) -> int:
        """
        Get the number of the selectors.

        Returns:
            int: number of the selectors.
        """        
        raise NotImplementedError()

class ValuesSelector(ConstantSelector):
    """
    Selector for constant values.

    Args:
        values (List[int]): List of values.
    """    
    def __init__(self, values: List[int]) -> None:
        self.values = values
    def get_values(self) -> List:
        return self.values
    def get_size(self) -> int:
        return len(self.values)

class ZeroSelector(ValuesSelector):
    """
    Selector for zeros.

    Args:
        size (int): Size of the selector.
    """    
    def __init__(self, size: int) -> None:
        super().__init__([0] * size)
        
class ByteSelector(ValuesSelector):
    """
    Selector for byte values.

    Args:
        size (int): Size of the selector.
        zero (bool, optional): Whether to include zero. Defaults to False.
    """    
    def __init__(self, size: int, zero=False) -> None:
        super().__init__([-128 / 2**i for i in range(0, size if not zero else size - 1)] + ([0] if zero else []) )
        self._zero  = zero

