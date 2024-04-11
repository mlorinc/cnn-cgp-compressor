import torch
import operator
from functools import reduce
from typing import List
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.base import BaseModel
from experiments.experiment import dequantize_per_tensor, conv2d_selector, FilterSelector
from experiments.multi_experiment import MultiExperiment

class SingleFilterZeroOutterExperiment(MultiExperiment):
    def __init__(self, config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, layer_names = ["conv1", "conv2"], prefix="") -> None:
        super().__init__(config, model, cgp, dtype)        

        for layer_name in layer_names:
            layer = getattr(self._model, layer_name)
            for i in range(layer.out_channels):
                for j in range(layer.in_channels):
                    sel = self._get_filter(layer_name, i, j)
                    self.zero_outter(sel)
                    self.create_experiment(f"{prefix}{layer_name}_{i}_{j}", sel)

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)

    def zero_outter(self, sel: FilterSelector):
        bias = self._get_bias(sel.layer_name)
        fp32_weights = self._get_reconstruction_weights(sel.layer_name)
        for output_selector in sel.out:
            w = fp32_weights[*output_selector]
            size = reduce(operator.mul, w.shape)
            fp32_weights[*output_selector] = dequantize_per_tensor(torch.zeros(size), w.q_scale(), w.q_zero_point())
        self._set_weights_bias(getattr(self._model, sel.layer_name), fp32_weights, bias)        

def init(config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, **args):
    return SingleFilterZeroOutterExperiment(config, model, cgp, dtype, **args)
