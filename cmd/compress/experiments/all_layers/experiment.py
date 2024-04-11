import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.base import BaseModel
from experiments.experiment import conv2d_core, conv2d_outter, conv2d_selector
from experiments.multi_experiment import MultiExperiment
from typing import List

class AllLayersExperiment(MultiExperiment):
    def __init__(self, config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, layer_names: List[str] = ["conv1", "conv2"], prefix="") -> None:
        super().__init__(config, model, cgp, dtype)
        self.mse_thresholds = [250, 150, 100, 50, 25]

        layers = [getattr(self._model, layer_name) for layer_name in layer_names]
        single_cell_size = 5 * max(map(lambda x: x.out_channels, layers))

        for mse in self.mse_thresholds:
            experiment = self.create_experiment(f"{prefix}mse_{mse}", self._get_filter(layer_names))
            experiment.config.set_row_count(single_cell_size)
            experiment.config.set_col_count(single_cell_size)
            experiment.config.set_look_back_parameter(single_cell_size)

    def _get_filters(self, layer_names: List[str]):
        out = []
        for layer_name in layer_names:
            out.append(conv2d_selector(layer_name, [slice(None), slice(None)], 5, 3))

def init(config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, **args):
    return AllLayersExperiment(config, model, cgp, dtype, **args)

