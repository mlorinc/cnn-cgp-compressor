import argparse
import torch
import random
from pathlib import Path
from typing import Dict, Generator, List, Self, Tuple, Union
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from models.quantization import conv2d_selector
from experiments.multi_experiment import MultiExperiment
from functools import partial

class GridSizeExperiment(MultiExperiment):
    name = "grid_size"
    default_grid_sizes=[(2, 2),(3, 3),(5, 5),(10, 10)]
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                args,
                grid_sizes=default_grid_sizes,
                layer_names=["conv1", "conv2"],
                prefix="",
                suffix="",
                dtype=torch.int8,
                n: int = 1,
                automatic_creation: bool = True,
                **kwargs
                ) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype, **kwargs)    
        assert n % len(layer_names) == 0
        self.grid_sizes = grid_sizes
        self.layer_names = layer_names
        self.prefix = prefix
        self.suffix = suffix
        self.k = n // len(layer_names)
        self._prepare_filters(automatic_creation)
        
    def _prepare_filters(self, automatic_creation: bool):
        if automatic_creation:
            for layer_name in self.layer_names:
                layer = self._model_adapter.get_layer(layer_name)
                combinations = random.sample([(a, b) for b in range(layer.out_channels) for a in range(layer.in_channels)], k=self.k)
                for row, col in self.grid_sizes:
                    for inp, out in combinations:
                        for experiment in self.setup_experiment(layer_name, inp, out, row, col):
                            self.register_experiment(experiment)
            
    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)

    def setup_experiment(self, layer_name: str, inp: int, out: int, row: int, col: int, register: bool = False) -> Generator[Self, None, None]:
        layer_name = str(layer_name); inp = int(inp); out = int(out)
        row = int(row); col = int(col)
        
        for experiment in self.create_experiment(f"{self.prefix}{layer_name}_x_{out}_{inp}_{row}_{col}{self.suffix}", self._get_filter(layer_name, out, inp), register=register, name_fmt=self.name_fmt):
            experiment.config.set_row_count(row)
            experiment.config.set_col_count(col)
            experiment.config.set_look_back_parameter(col)
            yield experiment

    @classmethod
    def with_replication(clf, reference_path: Union[Path, str], name_fmt: str, path: Union[Path, str], model_adapter: ModelAdapter, cgp: CGP, args, **kwargs):
        instance = MultiExperiment(reference_path, model_adapter, cgp, args, name_fmt=name_fmt)
        desired_instance = clf(path, model_adapter, cgp, args, automatic_creation=False, **kwargs)
        for experiment in instance.get_experiments():
            data = experiment.decode_configuration()
            for _ in desired_instance.setup_experiment(data["layer_name"], data["inp"], data["out"], data["row"], data["col"], register=True):
                pass
        desired_instance._prepared = True
        return desired_instance
    
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
        cls = GridSizeExperiment if not args.reuse else partial(GridSizeExperiment.with_replication, args.reuse, args.name_format)
        return cls(config, model_adapter, cgp, args,
                                        grid_sizes=args.grid_sizes,
                                        layer_names=args.layer_names,
                                        suffix=args.suffix,
                                        prefix=args.prefix,
                                        n=args.n,
                                        batches=args.batches)
