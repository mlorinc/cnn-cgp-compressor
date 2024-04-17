import argparse
from functools import partial
import random
import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.multi_experiment import MultiExperiment
from experiments.grid_size.experiment import GridSizeExperiment
from models.quantization import conv2d_selector

class ReversedSingleFilterExperiment(GridSizeExperiment):
    name = "reversed_single_filter"
    default_grids = [(2, 2), (3, 3), (5, 5), (10, 10)]
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                args,
                automatic_creation: bool = True,
                **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, args, automatic_creation=automatic_creation, **kwargs)         

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        sel = super()._get_filter(layer_name, filter_i, channel_i)
        sel.inp, sel.out = sel.out, sel.inp
        return sel
    
    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("-n", default=5, type=int, help="Amount of filters to be tested")
        parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of CNN layer names")
        parser.add_argument("--grid-sizes", nargs="+", type=int, default=GridSizeExperiment.default_grid_sizes, help="List of grid sizes (rows, columns)")
        parser.add_argument("--reuse", type=str, default=None, help="Reuse experiment configuration from the other experiment")
        parser.add_argument("--name-format", type=str, default=None, help="Name format of the resued experiments")
        MultiExperiment.get_argument_parser(parser)
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        cls = ReversedSingleFilterExperiment if not args.reuse else partial(ReversedSingleFilterExperiment.with_replication, args.reuse, args.name_format)
        return cls(config, model_adapter, cgp, args,
                                        grid_sizes=args.grid_sizes,
                                        layer_names=args.layer_names,
                                        suffix=args.suffix,
                                        prefix=args.prefix,
                                        n=args.n,
                                        batches=args.batches)