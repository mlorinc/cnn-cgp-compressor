import torch
from typing import List, Tuple
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.base import BaseModel
from experiments.experiment import BaseExperiment, conv2d_core, conv2d_outter

class GridSizeExperiment(BaseExperiment):
    name = "grid_size"
    def __init__(self, 
                experiment_folder: str, 
                model: BaseModel, 
                cgp: CGP, 
                dtype=torch.int8, experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or f"{GridSizeExperiment.name}/results", model, cgp, dtype)
        self.grid_sizes = [(5, 5), (10, 10), (15, 15), (20, 20), (30, 30), (50, 50)]
        self.filters = [
            *[(i, j, self._get_filter("conv1", i, j)) for j in range(model.conv1.in_channels) for i in range(model.conv1.out_channels)],
            *[(i, j, self._get_filter("conv2", i, j)) for j in range(model.conv2.in_channels) for i in range(model.conv2.out_channels)]
        ]

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return layer_name, [conv2d_core([filter_i, channel_i], 5, 3)], [*conv2d_outter([filter_i, channel_i], 5, 3)]

    def execute(self):
        for row, col in self.grid_sizes:
            for i, j, sel in self.filters:
                with self.new_experiment(f"{GridSizeExperiment.name}/results/{sel[0]}_{i}_{j}_{row}_{col}"):
                    try:
                        print(f"training: {self.experiment_root_path.name}")
                        self._cgp.config.set_row_count(row)
                        self._cgp.config.set_col_count(col)
                        self._cgp.config.set_look_back_parameter(col)
                        self.add_filters(*sel)
                        super().execute()
                    except FileExistsError:
                        print(f"skipping {sel[0]}_{i}_{j}_{row}_{col}")
                        continue

def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return GridSizeExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
