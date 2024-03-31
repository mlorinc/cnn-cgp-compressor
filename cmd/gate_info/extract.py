import glob
import re
import pandas as pd
import csv
import numpy as np
from pathlib import Path
from typing import Dict

key_order = {
    "add": 0,
    "bit_and": 1,
    "bit_dec": 2,
    "bit_inc": 3,
    "bit_lshift": 4,
    "bit_neg": 5,
    "bit_or": 6,
    "bit_rshift": 7,
    "bit_xor": 8,
    "clip": 9,
    "mul": 10,
    "neg": 11,
    "noop": 12,
    "reverse_max_a": 13,
    "reverse_max_b": 14,
    "reverse_min_a": 15,
    "reverse_min_b": 16,
    "reverse_mul_a": 17,
    "reverse_mul_b": 18,
    "reverse_mul_c": 19,
    "reverse_mul_d": 20,
    "sub": 21
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

    def save(self, df: pd.DataFrame):
        df.to_csv(self.output_file, header=True, index=True)

        df["order"] = df.index.map(key_order)
        df.sort_values("order", inplace=True)
        df.drop("order", axis=1, inplace=True)
        template = "{power} {delay} {area} {energy}"

        df_string: pd.Series = df.apply(lambda x: template.format(**x), 1)
        with open(self.output_file_text, "w") as f:
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
                power_values.append(segments[-2])
                power_units.append(segments[-1])
                index.append(file_name[:name_stop])

        df = pd.DataFrame({"power": power_values, "power_unit": power_units}, index=index)
        df["power"] = df["power"].astype("float")
        return df

    def _extract_delays(self) -> pd.DataFrame:
        values = []
        index = []
        for file in self._get_timing_files():
            with open(file, "r") as f:
                file_name = Path(file).name
                total_line = f.readlines()[-6].strip()
                segments = re.split("\s+", total_line)

                if " ".join(segments[:3]) != "data arrival time":
                    raise ValueError(f'[{file_name}] expecting "data arrival time" in the beginning of the line: "{total_line}"')
                if len(segments) != 4:
                    raise ValueError(f"[{file_name}] expected 4 segments; got: {len(segments)}")

                file_name = Path(file).name
                name_stop = file_name.rfind("_")
                values.append(segments[-1])
                index.append(file_name[:name_stop])

        return pd.DataFrame(values, index=index, columns=["delay"], dtype="float")
    
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
        powers = self._extract_powers()
        delays = self._extract_delays()
        areas = self._extract_areas()
        df = powers.join(delays).join(areas)

        if (df["power_unit"] != "mW").any():
            raise ValueError("incompatible power units")

        df["energy"] = df["power"] * df["delay"]
        df["energy_unit"] = "uJ"
        return df
        