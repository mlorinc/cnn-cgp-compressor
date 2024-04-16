import argparse
import torch
import operator
from functools import reduce
import random
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.multi_experiment import MultiExperiment
from models.selector import FilterSelector
from models.quantization import conv2d_selector, dequantize_per_tensor

class SingleFilterZeroOutterExperiment(MultiExperiment):
    name = "single_filter_zero_outter"

    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP,
                 args,
                 dtype=torch.int8, 
                 layer_names=["conv1", "conv2"], 
                 prefix="",
                 suffix="") -> None:
        super().__init__(config, model_adapter, cgp, args, dtype)

        zero_mse_experiments = []
        other_mse_experiments = []

        if model_adapter is not None:
            for layer_name in layer_names:
                layer = self._model_adapter.get_layer(layer_name)
                for i in range(layer.out_channels):
                    for j in range(layer.in_channels):
                        sel = self._get_filter(layer_name, i, j)
                        self.zero_outter(sel)
                        has_zero = self.has_zero_in_input(sel)
                        zero_suffix = "_zero" if has_zero else ""
                        experiment = self.create_experiment(f"{prefix}{layer_name}_{i}_{j}{zero_suffix}{suffix}", sel, register=False)
                        
                        if has_zero:
                            experiment.config.set_gate_count_early_stop(0)
                            zero_mse_experiments.append(experiment)
                        else:
                            experiment.config.set_gate_count_early_stop(1)
                            other_mse_experiments.append(experiment)
                            
        selecter_other_experiments = random.sample(other_mse_experiments, k=len(zero_mse_experiments))
        
        for a, b in zip(selecter_other_experiments, zero_mse_experiments):
            self.register_experiment(a)
            self.register_experiment(b)

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)

    def zero_outter(self, sel: FilterSelector):
        bias = self._model_adapter.get_bias(sel.layer_name)
        fp32_weights = self._model_adapter.get_weights(sel.layer_name)
        for output_selector in sel.out:
            w = fp32_weights[*output_selector]
            size = reduce(operator.mul, w.shape)
            fp32_weights[*output_selector] = dequantize_per_tensor(torch.zeros(size), w.q_scale(), w.q_zero_point())
        self._model_adapter.set_weights_bias(sel.layer_name, fp32_weights, bias)        

    def has_zero_in_input(self, sel: FilterSelector):
        fp32_weights = self._model_adapter.get_train_weights(sel.layer_name)
        for input_sel in sel.inp:
            w: torch.Tensor = fp32_weights[*input_sel]
            if torch.any(w == 0):
                return True
        return False

    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of CNN layer names")
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        MultiExperiment.get_argument_parser(parser)
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return SingleFilterZeroOutterExperiment(config, model_adapter, cgp, args, prefix=args.prefix, suffix=args.suffix)
