from decimal import Decimal
import pandas as pd
import math
from typing import Any, Dict, Tuple, Optional, Union, Iterable
from decimal import *
from functools import reduce
from operator import mul
import copy

class Quantizier(object):
    """
    A class to perform quantization and dequantization of values.

    Attributes:
        min_value (Decimal): Minimum value for quantization.
        max_value (Decimal): Maximum value for quantization.
        size (int): Size of the quantization grid.
        error_allowance (Decimal): Allowed error margin for quantization.
        bits (int): Number of bits required for quantization.
        max_error (Decimal): Maximum quantization error.
        _quant_max (int): Maximum quantized value.
    """    
    def __init__(self, min_value: float, max_value: float, grid_size: Tuple[int, ...] = None, axis: Union[Tuple[int, ...], int] = None, error_allowance: float = 1e-12) -> None:
        """
        Initializes the Quantizier with specified parameters.

        Args:
            min_value (float): Minimum value for quantization.
            max_value (float): Maximum value for quantization.
            grid_size (Tuple[int, ...], optional): Size of the quantization grid. Defaults to None.
            axis (Union[Tuple[int, ...], int], optional): Axis for quantization. Defaults to None.
            error_allowance (float, optional): Allowed error margin for quantization. Defaults to 1e-12.
        """        
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
        """
        Quantizes a value.

        Args:
            value (Union[Decimal, float]): The value to be quantized.

        Returns:
            int: The quantized value.
        """        
        result = int((Decimal(value) - self.min_value) / (self.max_value - self.min_value) * Decimal(self._quant_max))
        return result
    
    def dequantize(self, value: Union[Decimal, float]) -> float:
        """
        Dequantizes a value.

        Args:
            value (Union[Decimal, float]): The value to be dequantized.

        Returns:
            float: The dequantized value.
        """        
        result = (Decimal(value) / self._quant_max) * (self.max_value - self.min_value) + self.min_value
        return result

class QuantiziedSeries(Quantizier):
    """
    A class to perform quantization and dequantization of pandas Series.

    Attributes:
        data (pd.Series): The data series to be quantized.
    """    
    def __init__(self, data: pd.Series, grid_size: Tuple[int, ...], error_allowance: float = 1e-12, axis: Union[Tuple[int, ...], int] = None, min_value: float = None, max_value: float = None) -> None:
        """
        Initializes the QuantiziedSeries with specified parameters.

        Args:
            data (pd.Series): The data series to be quantized.
            grid_size (Tuple[int, ...]): Size of the quantization grid.
            error_allowance (float, optional): Allowed error margin for quantization. Defaults to 1e-12.
            axis (Union[Tuple[int, ...], int], optional): Axis for quantization. Defaults to None.
            min_value (float, optional): Minimum value for quantization. Defaults to data.min().
            max_value (float, optional): Maximum value for quantization. Defaults to data.max().
        """        
        super().__init__(min_value or data.min(), max_value or data.max(), grid_size, error_allowance=error_allowance, axis=axis) 
        self.data = data
    
    def quantize(self, value: Optional[Union[Decimal, float]] = None) -> Union[pd.Series, Decimal]:
        """
        Quantizes the series or a single value.

        Args:
            value (Optional[Union[Decimal, float]], optional): The value to be quantized. Defaults to None.

        Returns:
            Union[pd.Series, Decimal]: The quantized series or value.
        """        
        if value is None:
            self.data = self.data.apply(self.quantize)
            return self
        else:
            return super().quantize(value)
        
    def dequantize(self, value: Optional[int] = None, inline=False) -> Union[pd.Series, float]:
        """
        Dequantizes the series or a single value.

        Args:
            value (Optional[int], optional): The value to be dequantized. Defaults to None.
            inline (bool, optional): If True, dequantize the series inline. Defaults to False.

        Returns:
            Union[pd.Series, float]: The dequantized series or value.
        """        
        if value is None:
            return self.data.apply(self.dequantize, inline=inline)
        else:
            return super().dequantize(value)        
    
class DataframeQuantizier(pd.DataFrame):
    """
    A class to perform quantization and dequantization of pandas DataFrame.

    Attributes:
        grid_size (Tuple[int, int]): Size of the quantization grid.
    """    
    def __init__(self, data, grid_size: Tuple[int, int] = None, index=None, columns=None, dtype=None, copy=None) -> None:
        """
        Initializes the DataframeQuantizier with specified parameters.

        Args:
            data: DataFrame data.
            grid_size (Tuple[int, int], optional): Size of the quantization grid. Defaults to None.
            index: DataFrame index.
            columns: DataFrame columns.
            dtype: Data type.
            copy: Copy data.
        """        
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.grid_size = grid_size
       
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute of the DataframeQuantizier.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """        
        if name in ["grid_size"]:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)
        
    def __getattribute__(self, name: str) -> Any:
        """
        Gets an attribute of the DataframeQuantizier.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The value of the attribute.
        """        
        if name in ["grid_size"]:
            return self.__dict__[name]
        else:
            return super().__getattribute__(name)       
       
    def _get_column_mapping(self, columns: Optional[Union[Dict[str, str], Iterable[str], str]] = None) -> Dict[str, str]:
        """
        Gets the column mapping for quantization.

        Args:
            columns (Optional[Union[Dict[str, str], Iterable[str], str]], optional): The columns to be mapped. Defaults to None.

        Returns:
            Dict[str, str]: The column mapping.
        """        
        if isinstance(columns, Dict):
            return columns
        elif isinstance(columns, Iterable):
            return dict([(col, col) for col in columns])
        elif isinstance(columns, str):
            return {columns: columns}
        else:
            raise TypeError("invalid type")                 
        
    def quantize(self, columns: Optional[Union[Dict[str, str], Iterable[str], str]] = None, inline=False, error_allowance: float = 1e-12, axis: Union[Tuple[int, ...], int] = None) -> Union[pd.DataFrame, QuantiziedSeries]:
        """
        Quantizes the specified columns of the DataFrame.

        Args:
            columns (Optional[Union[Dict[str, str], Iterable[str], str]], optional): The columns to be quantized. Defaults to None.
            inline (bool, optional): If True, quantize inline. Defaults to False.
            error_allowance (float, optional): Allowed error margin for quantization. Defaults to 1e-12.
            axis (Union[Tuple[int, ...], int], optional): Axis for quantization. Defaults to None.

        Returns:
            Union[pd.DataFrame, QuantiziedSeries]: The quantized DataFrame or series.
        """        
        columns = self._get_column_mapping(columns)
        for target, src in columns.items():
            if inline:
                series = QuantiziedSeries(self[src], self.grid_size, error_allowance=error_allowance, axis=axis).quantize()
                self[target] = series.data
                return series
            else:
                return QuantiziedSeries(self[src], self.grid_size, error_allowance=error_allowance, axis=axis).quantize()
        