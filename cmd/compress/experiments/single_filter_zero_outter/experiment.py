import torch
import operator
from functools import reduce
from cgp.cgp_adapter import CGP
from models.base import BaseModel
from experiments.experiment import BaseExperiment, conv2d_core, conv2d_outter, dequantize_per_tensor

class SingleFilterZeroOutterExperiment(BaseExperiment):
    name = "single_filter_zero_outter"
    def __init__(self, experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, layer_name = "conv1", experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or SingleFilterZeroOutterExperiment.name, model, cgp, dtype)
        output_selectors = conv2d_outter([0, 0], 5, 3)
        self.add_filters(layer_name, 
            [
                conv2d_core([0, 0], 5, 3)
            ], 
            [
                *output_selectors
            ])
        
        bias = self._get_bias(layer_name)
        fp32_weights = self._get_reconstruction_weights(layer_name)
        for output_selector in output_selectors:
            w = fp32_weights[*output_selector]
            size = reduce(operator.mul, w.shape)
            fp32_weights[*output_selector] = dequantize_per_tensor(torch.zeros(size), w.q_scale(), w.q_zero_point())
        self._set_weights_bias(getattr(model, layer_name), fp32_weights, bias)


def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return SingleFilterZeroOutterExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
