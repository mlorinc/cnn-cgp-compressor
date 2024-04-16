import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.multi_experiment import MultiExperiment
from models.quantization import conv2d_selector
import argparse
from typing import List

class AllLayersExperiment(MultiExperiment):
    name = "all_layers"
    thresholds = [250, 150, 100, 50, 25, 15, 10, 0]
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP,
                 args,
                 dtype=torch.int8, 
                 layer_names=["conv1", "conv2"], 
                 mse_thresholds=thresholds,
                 prefix="",
                 suffix="",
                 rows_per_filter=5,
                 cols_per_layer=15) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype)
        
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.layer_names = layer_names
        self.rows_per_filter = rows_per_filter
        self.cols_per_layer = cols_per_layer

        layers = [self._model_adapter.get_layer(layer_name) for layer_name in layer_names]
        single_cell_size = self.rows_per_filter * max(map(lambda x: x.out_channels, layers))

        for mse in self.mse_thresholds:
            experiment = self.create_experiment(f"{prefix}mse_{mse}{suffix}", self._get_filters(layer_names))
            experiment.config.set_row_count(single_cell_size)
            experiment.config.set_col_count(self.cols_per_layer * len(layers))
            experiment.config.set_look_back_parameter(self.cols_per_layer * len(layers))

    def _get_filters(self, layer_names: List[str]):
        out = []
        for layer_name in layer_names:
            out.append(conv2d_selector(layer_name, [slice(None), slice(None)], 5, 3))
        return out

    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of CNN layer names")
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("--mse-thresholds", nargs="+", type=int, default=AllLayersExperiment.thresholds, help="List of MSE thresholds")
        parser.add_argument("--rows-per-filter", type=int, default=5, help="Number of rows per filter")
        parser.add_argument("--cols-per-layer", type=int, default=15, help="Number of columns per layer")
        MultiExperiment.get_argument_parser(parser)
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return AllLayersExperiment(config, model_adapter, cgp, args,
                                   layer_names=args.layer_names,
                                   prefix=args.prefix,
                                   suffix=args.suffix,
                                   mse_thresholds=args.mse_thresholds,
                                   rows_per_filter=args.rows_per_filter,
                                   cols_per_layer=args.cols_per_layer)
