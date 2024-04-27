import argparse
from functools import partial
import random
import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.composite.experiment import MultiExperiment
from experiments.grid_size.experiment import GridSizeExperiment
from models.quantization import conv2d_selector

class ReversedSingleFilterExperiment(GridSizeExperiment):
    name = "reversed_single_filter"
    default_grids = [(2, 2), (3, 3), (5, 5), (10, 10)]
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                automatic_creation: bool = True,
                **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, automatic_creation=automatic_creation, **kwargs)         

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        sel = super()._get_filter(layer_name, filter_i, channel_i)
        sel.inp, sel.out = sel.out, sel.inp
        return sel
    
    @classmethod
    def with_cli_arguments(cls, config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        cls = cls if not args.reuse else partial(ReversedSingleFilterExperiment.with_replication, args.reuse, args.name_format)
        return cls(config, model_adapter, cgp, **args)