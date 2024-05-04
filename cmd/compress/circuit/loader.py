import os
from circuit.extract import DataExtractor
from decimal import *
from typing import Optional, Tuple, Union
from pathlib import Path
from circuit.quantizer import DataframeQuantizier
from commands.datastore import Datastore

def get_gate_parameters(csv_file: Union[Path, str], txt_file: Union[Path, str], grid_size: Tuple[int, int] = None, quant_bits=64, data_dir: Optional[Union[Path, str]] = None):
    txt_file = Path(txt_file)
    csv_file = Path(csv_file)
    data_dir = os.environ.get("gate_parameters_dir", Datastore().derive("verilog"))
    extractor = DataExtractor(data_dir)
    df = DataframeQuantizier(data=extractor.extract(), grid_size=grid_size)
    energy_series = df.quantize(columns={"quantized_energy": "energy"}, inline=True)
    delay_series = df.quantize(columns={"quantized_delay": "delay"}, axis=1, inline=True)
    extractor.save(df, csv_file, txt_file)
    return df, energy_series, delay_series
