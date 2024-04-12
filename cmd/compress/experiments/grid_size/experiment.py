import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from models.quantization import conv2d_selector
from experiments.multi_experiment import MultiExperiment
import argparse

class GridSizeExperiment(MultiExperiment):
    name = "grid_size"
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP, 
                grid_sizes=[(5, 5), (10, 10)],
                layer_names=["conv1", "conv2"],
                prefix="",
                suffix="",
                dtype=torch.int8) -> None:
        super().__init__(config, model_adapter, cgp, dtype)
        self.grid_sizes = grid_sizes
        self.layer_names = layer_names

        for layer_name in layer_names:
            layer = self._model_adapter.get_layer(layer_name)
            for row, col in self.grid_sizes:
                for i in range(layer.out_channels):
                    for j in range(layer.in_channels):
                        experiment = self.create_experiment(f"{prefix}{layer_name}_{i}_{j}_{row}_{col}{suffix}", self._get_filter(layer_name, i, j))
                        experiment.config.set_row_count(row)
                        experiment.config.set_col_count(col)
                        experiment.config.set_look_back_parameter(col)

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)

    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of CNN layer names")
        parser.add_argument("--grid-sizes", nargs="+", type=int, default=[(5, 5), (10, 10)], help="List of grid sizes (rows, columns)")
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
        return GridSizeExperiment(config, model_adapter, cgp,
                                        grid_sizes=args.grid_sizes,
                                        layer_names=args.layer_names,
                                        suffix=args.suffix,
                                        prefix=args.prefix)
