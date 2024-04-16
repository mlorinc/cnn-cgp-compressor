import argparse
import random
import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.multi_experiment import MultiExperiment
from models.quantization import conv2d_selector

class ReversedSingleFilterExperiment(MultiExperiment):
    name = "reversed_single_filter"
    default_grids = [(2, 2), (3, 3), (5, 5), (10, 10)]
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                args,
                grid_sizes=default_grids,
                layer_names=["conv2"],
                suffix="",
                dtype=torch.int8, prefix="", n: int = 1) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype)
        assert n % len(layer_names) == 0
        self.grid_sizes = grid_sizes
        self.layer_names = layer_names
        self.suffix = suffix
        self.k = n // len(layer_names)
        normalized_patience = self.config.get_patience() // 5

        for layer_name in layer_names:
            layer = self._model_adapter.get_layer(layer_name)
            assert self.k <= layer.in_channels and self.k <= layer.out_channels
            in_indices = random.sample(range(layer.in_channels), k=self.k)
            out_indices = random.sample(range(layer.out_channels), k=self.k)
            for row, col in self.grid_sizes:
                for i, j in zip(in_indices, out_indices):
                    experiment = self.create_experiment(f"{prefix}{layer_name}_{i}_{j}_{row}_{col}{suffix}", self._get_filter(layer_name, j, i))
                    experiment.config.set_row_count(row)
                    experiment.config.set_col_count(col)
                    experiment.config.set_patience(normalized_patience * row)
                    experiment.config.set_look_back_parameter(col)             

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        sel = conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)
        sel.inp, sel.out = sel.out, sel.inp
        return sel
    
    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("-n", default=5, type=int, help="Amount of filters to be tested")
        parser.add_argument("--layer-names", nargs="+", default=["conv2"], help="List of CNN layer names")
        parser.add_argument("--grid-sizes", nargs="+", type=int, default=[(2, 2), (3, 3), (5, 5), (10, 10)], help="List of grid sizes (rows, columns)")
        MultiExperiment.get_argument_parser(parser)
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return ReversedSingleFilterExperiment(config, model_adapter, cgp, args,
                                        grid_sizes=args.grid_sizes,
                                        layer_names=args.layer_names,
                                        suffix=args.suffix,
                                        prefix=args.prefix,
                                        n=args.n)
