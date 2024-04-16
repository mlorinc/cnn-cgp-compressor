import glob
import re
import pandas as pd
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union
from decimal import *
import math

key_order = {
    "noop": 0,
    "reverse_max_a": 1,
    "add": 2,
    "sub": 3,
    "mul": 4,
    "neg": 5,
    "reverse_min_b": 6,
    "quarter": 7,
    "half": 8,
    "bit_and": 9,
    "bit_or": 10,
    "bit_xor": 11,
    "bit_neg": 12,
    "double": 13,
    "bit_inc": 14,
    "bit_dec": 15,
    "r_shift_3": 16,
    "r_shift_4": 17,
    "r_shift_5": 18,
    "l_shift_2": 19,
    "l_shift_3": 20,
    "l_shift_4": 21,
    "l_shift_5": 22,
    "one_const": 23,
    "minus_one_const": 24,
    "zero_const": 25,
    "expected_value_min": 26,
    "expected_value_max": 27,
    "mux_2to1": 28,
    "mux_4to1": 29,
    "mux_8to1": 30,
    "mux_16to1": 31,
    "demux_2to1": 32,
    "demux_4to1": 33,
    "demux_8to1": 34,
    "demux_16to1": 35,
}

class DataExtractor(object):
    def __init__(self, basedir: str) -> None:
        self.basedir = Path(basedir)
        self.input_dir = self.basedir / "res"
        self.output_file = self.basedir / "parameters.csv"
        self.output_file_text = self.basedir / "parameters.txt"
    
    def _get_power_files(self):
        return glob.glob(str(self.input_dir / "*_power.txt"))

    def _get_area_files(self):
        return glob.glob(str(self.input_dir / "*_area.txt"))
    
    def _get_timing_files(self):
        return glob.glob(str(self.input_dir / "*_timing.txt"))

    def save(self, df: pd.DataFrame, output_csv: Union[str], output_txt: Union[str]):
        df.to_csv(output_csv or self.output_file, header=True, index=True)

        df = df.loc[df.index.isin(key_order.keys()), :].copy()
        df["order"] = df.index.map(key_order)
        df.sort_values("order", inplace=True)
        df.drop("order", axis=1, inplace=True)
        template = "{quantized_energy} {energy} {area} {quantized_delay} {delay}"

        df_string: pd.Series = df.apply(lambda x: template.format(**x), 1)
        with open(output_txt or self.output_file_text, "w", newline="\n") as f:
            f.writelines(line + "\n" for line in df_string)
        

    def _extract_powers(self) -> pd.DataFrame:
        power_values = []
        power_units = []
        index = []
        for file in self._get_power_files():
            with open(file, "r") as f:
                file_name = Path(file).name
                total_line = f.readlines()[-2].strip()
                segments = re.split("\s+", total_line)

                if segments[0] != "Total":
                    raise ValueError(f'[{file_name}] expecting "Total" in the beginning of the line: "{total_line}"')
                if len(segments) != 9:
                    raise ValueError(f"[{file_name}] expected 9 segments; got: {len(segments)}")
                if any([not segments[i].endswith("W") for i in range(2, len(segments), 2)]):
                    raise ValueError(f'[{file_name}] unexpected format of the power line: "{total_line}"')

                file_name = Path(file).name
                name_stop = file_name.rfind("_")
                power_values.append(Decimal(segments[-2]))
                power_units.append(segments[-1])
                index.append(file_name[:name_stop])

        df = pd.DataFrame({"power": power_values, "power_unit": power_units}, index=index)
        return df

    def _extract_delays(self) -> pd.DataFrame:
        values = []
        index = []
        for file in self._get_timing_files():
            with open(file, "r") as f:
                file_name = Path(file).name
                name_stop = file_name.rfind("_")
                lines = f.readlines()
                total_line = lines[-6].strip()
                no_paths_line = lines[-3].strip()
                segments = re.split("\s+", total_line)

                if no_paths_line == "No paths.":
                    values.append(0)
                    index.append(file_name[:name_stop])
                    continue

                if " ".join(segments[:3]) != "data arrival time":
                    raise ValueError(f'[{file_name}] expecting "data arrival time" in the beginning of the line: "{total_line}"')
                if len(segments) != 4:
                    raise ValueError(f"[{file_name}] expected 4 segments; got: {len(segments)}")

                values.append(Decimal(segments[-1]))
                index.append(file_name[:name_stop])

        return pd.DataFrame(values, index=index, columns=["delay"])
    
    def _extract_areas(self) -> pd.DataFrame:
        values = []
        index = []
        for file in self._get_area_files():
            with open(file, "r") as f:
                file_name = Path(file).name
                total_line = f.readlines()[-3].strip()
                segments = re.split("\s+", total_line)

                if " ".join(segments[:3]) != "Total cell area:":
                    raise ValueError(f'[{file_name}] expecting "Total cell area:" in the beginning of the line: "{total_line}"')
                if len(segments) != 4:
                    raise ValueError(f"[{file_name}] expected 4 segments; got: {len(segments)}")

                name_stop = file_name.rfind("_")
                values.append(segments[-1])
                index.append(file_name[:name_stop])

        return pd.DataFrame(values, index=index, columns=["area"], dtype="float")

    def extract(self):
        getcontext().prec = 32
        powers = self._extract_powers()
        delays = self._extract_delays()
        areas = self._extract_areas()
        df = powers.join(delays).join(areas)

        if (df["power_unit"] != "mW").any():
            raise ValueError("incompatible power units")

        df["energy"] = df["power"] * df["delay"]
        df["energy_unit"] = "uJ"
        
        return df

def default_value_estimator(grid: Tuple[int, int]) -> Decimal:
    return Decimal(grid[0] * grid[1])

def delay_value_estimator(grid: Tuple[int, int]) -> Decimal:
    return Decimal(grid[1])

def quantize_series(data: pd.Series, grid_size: Tuple[int, int], quant_bits: int=8, error_max: float = 1e-6, inclue_metrics: bool = False, largest_value_estimator = default_value_estimator):
        getcontext().prec = 32
        error_max = Decimal(error_max)
        assert isinstance(data.iloc[0], Decimal)
        e_min, e_max = Decimal(0) * largest_value_estimator(grid_size), Decimal(math.ceil(data.max() * largest_value_estimator(grid_size)))
        # First -1 because of the zero and the second one to reserve space for nan value in the CGP
        quant_max = int(Decimal(2**(quant_bits) - 1 - 1))
        # print(e_min, e_max, quant_max, quant(e_max, e_min, e_max, quant_max))
        quantized_series = data.apply(lambda x: quant(x, e_min, e_max, quant_max))
        dequantized_series = quantized_series.apply(lambda x: dequant(x, e_min, e_max, quant_max))
        errors = data - dequantized_series
        # print(quantized_series)
        if inclue_metrics:
            # max_error = 0
            # for num in drange(e_min, e_max, data.max()):
            #     x = quant(num, e_min, e_max, quant_max)
            #     dx = dequant(x, e_min, e_max, quant_max)
            #     error = abs(num - dx)
            #     max_error = error
            #     if error > error_max:
            #         print(num, error)
            #         raise ValueError("no solution found")    
            print("quant_bits:", quant_bits, "quant_error:", errors.max(), "max_error:", (e_max - e_min) / quant_max, "actual_max_error:", None)
            return quantized_series, e_min, e_max, quant_max
        else:
            return quantized_series, e_min, e_max, quant_max

def dequantize_series(data: pd.Series, grid_size: Tuple[int, int], quant_bits: int=8):
        getcontext().prec = 32
        grid_size = grid_size[0] * grid_size[1]
        assert isinstance(data.iloc[0], Decimal)
        e_min, e_max = data.min() * grid_size, data.max() * grid_size
        quant_max = Decimal(2**(quant_bits) - 1)
        return data.apply(lambda x: dequant(x, e_min, e_max, quant_max))

def drange(x, y, jump):
  while x < y:
    yield Decimal(x)
    x += Decimal(jump)    
        
def quant(x: Decimal, e_min: Decimal, e_max: Decimal, quant_max: Decimal) -> int:
    result = int((Decimal(x) - e_min) / (e_max - e_min) * Decimal(quant_max))
    return result

def dequant(x: int, e_min: Decimal, e_max: Decimal, quant_max: Decimal) -> Decimal:
    return (Decimal(x) / Decimal(quant_max)) * (e_max - e_min) + e_min