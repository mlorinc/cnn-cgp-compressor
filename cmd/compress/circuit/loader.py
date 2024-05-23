import os
from circuit.extract import DataExtractor
from decimal import *
from typing import Optional, Tuple, Union
from pathlib import Path
from circuit.quantizer import DataframeQuantizier
from commands.datastore import Datastore

def get_gate_parameters(csv_file: Union[Path, str], txt_file: Union[Path, str], grid_size: Tuple[int, int] = None, quant_bits=64, data_dir: Optional[Union[Path, str]] = None):
    """
    Extracts gate parameters, quantizes the data, and saves it to CSV and text files.

    Args:
        csv_file (Union[Path, str]): Path to the CSV file where the data will be saved.
        txt_file (Union[Path, str]): Path to the text file where the data will be saved.
        grid_size (Tuple[int, int], optional): Grid size for the quantizer. Defaults to None.
        quant_bits (int, optional): Number of quantization bits. Defaults to 64.
        data_dir (Optional[Union[Path, str]], optional): Directory containing the input data files. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: The DataFrame with extracted data, energy series, and delay series.
    """    
    txt_file = Path(txt_file)
    csv_file = Path(csv_file)
    data_dir = os.environ.get("gate_parameters_dir", Datastore().derive("verilog"))
    extractor = DataExtractor(data_dir)
    df = DataframeQuantizier(data=extractor.extract(), grid_size=grid_size)
    energy_series = df.quantize(columns={"quantized_energy": "energy"}, inline=True)
    delay_series = df.quantize(columns={"quantized_delay": "delay"}, axis=1, inline=True)
    extractor.save(df, csv_file, txt_file)
    return df, energy_series, delay_series
