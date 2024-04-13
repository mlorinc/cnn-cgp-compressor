import argparse
import torch
import operator
from functools import reduce
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
                 dtype=torch.int8, 
                 layer_names=["conv1", "conv2"], 
                 prefix="",
                 suffix="") -> None:
        super().__init__(config, model_adapter, cgp, dtype)        

        for layer_name in layer_names:
            layer = self._model_adapter.get_layer(layer_name)
            for i in range(layer.out_channels):
                for j in range(layer.in_channels):
                    sel = self._get_filter(layer_name, i, j)
                    self.zero_outter(sel)
                    experiment = self.create_experiment(f"{prefix}{layer_name}_{i}_{j}{suffix}", sel)
                    experiment.config.set_gate_count_early_stop(0 if self.has_zero_in_input(sel) else 1);

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
        return parser

    @staticmethod
    def get_pbs_argument_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--time-limit", required=True, help="Time limit for the PBS job")
        parser.add_argument("--template-pbs-file", required=True, help="Path to the template PBS file")
        parser.add_argument("--experiments-folder", default="experiments_folder", help="Experiments folder")
        parser.add_argument("--results-folder", default="results", help="Results folder")
        parser.add_argument("--cgp-folder", default="cgp_cpp_project", help="CGP folder")
        parser.add_argument("--cpu", type=int, default=32, help="Number of CPUs")
        parser.add_argument("--mem", default="2gb", help="Memory")
        parser.add_argument("--scratch-capacity", default="1gb", help="Scratch capacity")
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return SingleFilterZeroOutterExperiment(config, model_adapter, cgp, args, prefix=args.prefix, suffix=args.suffix)
