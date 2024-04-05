import torch
from cgp.cgp_adapter import CGP
from models.base import BaseModel
from experiments.experiment import conv2d_core, conv2d_outter
from experiments.multi_experiment import MultiExperiment

class AllLayersExperiment(MultiExperiment):
    name = "all_layers"
    def __init__(self, experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or AllLayersExperiment.name, model, cgp, dtype)
        self.mse_thresholds = [250, 150, 100, 50, 25]

    def _prepare_filters(self):
        self.add_filters("conv1",
                         [conv2d_core([slice(None), slice(None)], 5, 3)], 
                         [*conv2d_outter([slice(None), slice(None)], 5, 3)]
                         )
        self.next_input_combination()
        self.add_filters("conv2",
                         [conv2d_core([slice(None), slice(None)], 5, 3)], 
                         [*conv2d_outter([slice(None), slice(None)], 5, 3)]
                         )

    def execute(self):
        for mse_threshold in self.mse_thresholds:
            with self.experiment_context(f"{mse_threshold}") as (experiment_name, config):
                try:
                    config.set_row_count(5 * self._model.conv2.out_channels)
                    config.set_col_count(30)
                    config.set_look_back_parameter(15)
                    self._prepare_filters()
                    print(f"training: {experiment_name}")
                    config.set_mse_threshold(mse_threshold)
                    super().execute(config)
                except FileExistsError:
                    print(f"skipping {experiment_name}")
                    continue

def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return AllLayersExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
