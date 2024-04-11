import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.base import BaseModel
from experiments.experiment import conv2d_selector, FilterSelector
from experiments.multi_experiment import MultiExperiment

class GridSizeExperiment(MultiExperiment):
    def __init__(self, 
                config: CGPConfiguration,
                model: BaseModel, 
                cgp: CGP, 
                dtype=torch.int8, layer_names = ["conv1", "conv2"], prefix="") -> None:
        super().__init__(config, model, cgp, dtype)
        self.grid_sizes = [(5, 5), (10, 10)]

        for layer_name in layer_names:
            layer = getattr(self._model, layer_name)
            for row, col in self.grid_sizes:
                for i in range(layer.out_channels):
                    for j in range(layer.in_channels):
                        experiment = self.create_experiment(f"{prefix}{layer_name}_{i}_{j}_{row}_{col}", self._get_filter(layer_name, i, j))
                        experiment.config.set_row_count(row)
                        experiment.config.set_col_count(col)
                        experiment.config.set_look_back_parameter(col)

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)


def init(config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, **args):
    return GridSizeExperiment(config, model, cgp, dtype, **args)
