import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
import argparse

class SingleFilterExperiment(Experiment):
    name = "single_filter"
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                grid_size=(5, 5),
                layer_name="conv1",
                channel=0,
                filter=0,
                patience=100000,
                prefix="",
                suffix="",
                dtype=torch.int8) -> None:
        super().__init__(config, model_adapter, cgp, dtype)
        self.set_paths(self.base_folder / (prefix + self.base_folder.name + suffix))
        self.grid_size = grid_size
        self.layer_name = layer_name

        self.add_filter_selector(self._get_filter(layer_name, filter_i=filter, channel_i=channel))
        self.config.set_patience(patience)
        self.config.set_row_count(grid_size[0])
        self.config.set_col_count(grid_size[1])
        self.config.set_look_back_parameter(grid_size[1])

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)
    
    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("--layer-name", default="conv1", help="Name of the CNN layer")
        parser.add_argument("--channel", type=int, default=0, help="Channel index")
        parser.add_argument("--filter", type=int, default=0, help="Filter index")
        parser.add_argument("--grid-size", nargs=2, type=int, default=[5, 5], help="Grid size (rows, columns)")
        parser.add_argument("--patience", type=int, default=100000, help="Patience value")
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
        return SingleFilterExperiment(config, model_adapter, cgp,
                                        grid_size=args.grid_size,
                                        layer_name=args.layer_name,
                                        channel=args.channel,
                                        filter=args.filter,
                                        patience=args.patience,
                                        suffix=args.suffix,
                                        prefix=args.prefix)
