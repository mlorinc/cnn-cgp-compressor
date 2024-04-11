import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.base import BaseModel
from experiments.experiment import conv2d_selector
from experiments.multi_experiment import MultiExperiment

class SingleChannelExperiment(MultiExperiment):
    def __init__(self, config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, layer_name="conv1", channel=0, prefix="") -> None:
        super().__init__(config, model, cgp, dtype)
        
        self.mse_thresholds = [0, 50, 25, 15, 10, 5, 2, 1, 0.5]

        layer = getattr(self._model, layer_name)
        single_cell_size = 5 * layer.out_channels
        for mse in self.mse_thresholds:
            experiment = self.create_experiment(f"{prefix}{layer_name}_mse_{mse}", self._get_filter(layer_name, channel))
            experiment.config.set_mse_threshold(mse)
            experiment.config.set_row_count(single_cell_size)
            experiment.config.set_col_count(15)
            experiment.config.set_look_back_parameter(15)

    def _get_filter(self, layer_name: str, channel_i: int):
        return conv2d_selector(layer_name, [slice(None), channel_i], 5, 3)

def init(config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, **args):
    return SingleChannelExperiment(config, model, cgp, dtype, **args)
