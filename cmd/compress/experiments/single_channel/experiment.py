import torch
from cgp.cgp_adapter import CGP
from models.base import BaseModel
from experiments.experiment import conv2d_core, conv2d_outter
from experiments.multi_experiment import MultiExperiment

class SingleChannelExperiment(MultiExperiment):
    name = "single_channel"
    def __init__(self, experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or SingleChannelExperiment.name, model, cgp, dtype)
        
        single_cell_size = 5 * model.conv1.out_channels
        self._cgp.config.set_row_count(single_cell_size)
        self._cgp.config.set_col_count(15)
        self._cgp.config.set_look_back_parameter(15)
        self.mse_thresholds = [50, 25, 15, 10, 5, 2, 1, 0.7, 0]

    def execute(self):
        for mse_threshold in self.mse_thresholds:
            with self.experiment_context(f"{mse_threshold}") as (experiment_name, config):
                try:
                    self.add_filters("conv1", [conv2d_core([slice(None), 0], 5, 3)], [*conv2d_outter([slice(None), 0], 5, 3)])
                    print(f"training: {experiment_name}")
                    config.set_mse_threshold(mse_threshold)
                    super().execute(config)
                except FileExistsError:
                    print(f"skipping {experiment_name}")
                    continue

def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return SingleChannelExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
