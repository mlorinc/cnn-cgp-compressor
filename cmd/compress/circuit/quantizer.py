from decimal import Decimal
import pandas as pd
import math
from typing import Any, Dict, Tuple, Optional, Union, Iterable
from decimal import *
from functools import reduce
from operator import mul
import copy

class Quantizier(object):
    def __init__(self, min_value: float, max_value: float, grid_size: Tuple[int, ...] = None, axis: Union[Tuple[int, ...], int] = None, error_allowance: float = 1e-12) -> None:
        getcontext().prec = 32
        axis = axis if isinstance(axis, tuple) else tuple([axis]) if isinstance(axis, int) else range(len(grid_size))
        if grid_size is not None:
            grid_sizes = map(lambda x: grid_size[x], axis)
            self.size = reduce(mul, grid_sizes, 1)
        else:
            self.size = 1
        print(f"min value: {min_value}", f"max value: {max_value}")
        self.min_value = Decimal(min_value * self.size)
        self.max_value = Decimal(max_value * self.size)
        print(f"min total value: {self.min_value}", f"max total value: {self.max_value}")
        self.error_allowance = Decimal(error_allowance)
        print(f"error allowance: {self.error_allowance}")
        self.bits = int(math.ceil(math.log2((self.max_value - self.min_value) / self.error_allowance + 2)))
        self.max_error = (self.max_value - self.min_value) / (2**self.bits - 2)
        
        if self.max_error >= error_allowance:
            raise ValueError(f"cannot quantized because error is more significant than anticipated: {error_allowance} <= {self.max_error}")
        
        print("max quantization error:", self.max_error)
        for limit in [8, 16, 32, 64]:
            if self.bits <= limit:
                self.bits = limit
                break
        else:
            raise ValueError("Quantization requires more than 64 bits which is not supported by this implementation")
    
        self._quant_max = int(Decimal(2**(self.bits) - 1 - 1))
        
    def quantize(self, value: Union[Decimal, float]) -> int:
        result = int((Decimal(value) - self.min_value) / (self.max_value - self.min_value) * Decimal(self._quant_max))
        return result
    
    def dequantize(self, value: Union[Decimal, float]) -> int:
        result = (Decimal(value) / self._quant_max) * (self.max_value - self.min_value) + self.min_value
        return result

class QuantiziedSeries(Quantizier):
    def __init__(self, data: pd.Series, grid_size: Tuple[int, ...], error_allowance: float = 1e-12, axis: Union[Tuple[int, ...], int] = None, min_value: float = None, max_value: float = None) -> None:
        super().__init__(min_value or data.min(), max_value or data.max(), grid_size, error_allowance=error_allowance, axis=axis) 
        self.data = data
    
    def quantize(self, value: Optional[Union[Decimal, float]] = None) -> Union[pd.Series, Decimal]:
        if value is None:
            self.data = self.data.apply(self.quantize)
            return self
        else:
            return super().quantize(value)
        
    def dequantize(self, value: Optional[int] = None, inline=False) -> Union[pd.Series, int]:
        if value is None:
            return self.data.apply(self.dequantize, inline=inline)
        else:
            return super().dequantize(value)        
    
class DataframeQuantizier(pd.DataFrame):
    def __init__(self, data, grid_size: Tuple[int, int] = None, index=None, columns=None, dtype=None, copy=None) -> None:
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.grid_size = grid_size
       
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["grid_size"]:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)
        
    def __getattribute__(self, name: str) -> Any:
        if name in ["grid_size"]:
            return self.__dict__[name]
        else:
            return super().__getattribute__(name)       
       
    def _get_column_mapping(self, columns: Optional[Union[Dict[str, str], Iterable[str], str]] = None) -> Dict[str, str]:
        if isinstance(columns, Dict):
            return columns
        elif isinstance(columns, Iterable):
            return dict([(col, col) for col in columns])
        elif isinstance(columns, str):
            return {columns: columns}
        else:
            raise TypeError("invalid type")                 
        
    def quantize(self, columns: Optional[Union[Dict[str, str], Iterable[str], str]] = None, inline=False, error_allowance: float = 1e-12, axis: Union[Tuple[int, ...], int] = None) -> Union[pd.DataFrame, QuantiziedSeries]:
        columns = self._get_column_mapping(columns)
        for target, src in columns.items():
            if inline:
                series = QuantiziedSeries(self[src], self.grid_size, error_allowance=error_allowance, axis=axis).quantize()
                self[target] = series.data
                return series
            else:
                return QuantiziedSeries(self[src], self.grid_size, error_allowance=error_allowance, axis=axis).quantize()
        