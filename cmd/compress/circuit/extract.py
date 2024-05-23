import glob
import re
import pandas as pd
from pathlib import Path
from typing import Union
from decimal import *

# Dictionary to maintain order of keys for various operations
key_order = {
    "reverse_max_a": 0,
    "add": 1,
    "sub": 2,
    "mul": 3,
    "neg": 4,
    "reverse_min_b": 5,
    "quarter": 6,
    "half": 7,
    "bit_and": 8,
    "bit_or": 9,
    "bit_xor": 10,
    "bit_neg": 11,
    "double": 12,
    "bit_inc": 13,
    "bit_dec": 14,
    "r_shift_3": 15,
    "r_shift_4": 16,
    "r_shift_5": 17,
    "l_shift_2": 18,
    "l_shift_3": 19,
    "l_shift_4": 20,
    "l_shift_5": 21,
    "one_const": 22,
    "minus_one_const": 23,
    "zero_const": 24,
    "expected_value_min": 25,
    "expected_value_max": 26,
    "mux_2to1": 27,
    "mux_4to1": 28,
    "mux_8to1": 29,
    "mux_16to1": 30,
    "demux_2to1": 31,
    "demux_4to1": 32,
    "demux_8to1": 33,
    "demux_16to1": 34,
}

key_order_with_id = {**key_order, "id": 100}

class DataExtractor(object):
    """
    A class to extract power, area, and timing data from specified files and save the processed data.

    Attributes:
        basedir (Path): The base directory where the input files are located and output files will be saved.
        input_dir (Path): Directory containing the input files.
        output_file (Path): Path to the CSV file where the extracted data will be saved.
        output_file_text (Path): Path to the text file where the extracted data will be saved in text format.
    """    
    def __init__(self, basedir: str) -> None:
        """
        Initializes the DataExtractor with a base directory.

        Args:
            basedir (str): The base directory where input and output files are located.
        """        
        self.basedir = Path(basedir)
        self.input_dir = self.basedir / "res"
        self.output_file = self.basedir / "parameters.csv"
        self.output_file_text = self.basedir / "parameters.txt"
    
    def _get_power_files(self):
        """
        Retrieves the list of power files from the input directory.

        Returns:
            list: A list of file paths matching the pattern *_power.txt in the input directory.
        """        
        return glob.glob(str(self.input_dir / "*_power.txt"))

    def _get_area_files(self):
        """
        Retrieves the list of area files from the input directory.

        Returns:
            list: A list of file paths matching the pattern *_area.txt in the input directory.
        """        
        return glob.glob(str(self.input_dir / "*_area.txt"))
    
    def _get_timing_files(self):
        """
        Retrieves the list of timing files from the input directory.

        Returns:
            list: A list of file paths matching the pattern *_timing.txt in the input directory.
        """        
        return glob.glob(str(self.input_dir / "*_timing.txt"))

    def save(self, df: pd.DataFrame, output_csv: Union[str], output_txt: Union[str]):
        """
        Saves the extracted data to CSV and text files.

        Args:
            df (pd.DataFrame): The DataFrame containing the extracted data.
            output_csv (Union[str, Path]): Path to the CSV file where the data will be saved. Defaults to self.output_file.
            output_txt (Union[str, Path]): Path to the text file where the data will be saved. Defaults to self.output_file_text.
        """        
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
        """
        Extracts power data from power files.

        Returns:
            pd.DataFrame: A DataFrame containing power values and units.
        """        
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
        """
        Extracts delay data from timing files.

        Returns:
            pd.DataFrame: A DataFrame containing delay values.
        """        
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
        """
        Extracts area data from area files.

        Returns:
            pd.DataFrame: A DataFrame containing area values.
        """        
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
        """
        Extracts power, delay, and area data from files and combines them into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing combined power, delay, and area data.
        """        
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
