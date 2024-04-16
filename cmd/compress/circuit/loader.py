import pandas as pd
from circuit.extract import DataExtractor, quantize_series, delay_value_estimator
from decimal import *
from typing import Optional, Tuple, Union
from pathlib import Path


def get_gate_parameters(csv_file: Union[Path, str], txt_file: Union[Path, str], grid_size: Tuple[int, int] = None, quant_bits=64, data_dir: Optional[Union[Path, str]] = r"C:\Users\Majo\source\repos\TorchCompresser\cmd\compress\circuit\data"):
    extractor = DataExtractor(data_dir)
    df = extractor.extract()
    energies, e_min, e_max, quant_max = quantize_series(df["energy"], grid_size, quant_bits=quant_bits, inclue_metrics=False)
    delays, d_min, d_max, quant_max = quantize_series(df["delay"], grid_size, quant_bits=quant_bits, inclue_metrics=False, largest_value_estimator=delay_value_estimator)
    df["quantized_energy"] = energies
    df["quantized_delay"] = delays
    
    extractor.save(df, csv_file, txt_file)
    return df
