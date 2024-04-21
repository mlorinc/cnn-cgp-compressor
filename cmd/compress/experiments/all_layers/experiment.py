import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.multi_experiment import MultiExperiment
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
import argparse
from typing import List
from parse import parse

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
                 prepare=True,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype, **kwargs)
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.layer_names = layer_names
        
        if prepare:
            self._prepare_filters()

    def _prepare_filters(self):
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}mse_{mse}_{self.args.rows}_{self.args.cols}{self.suffix}", self._get_filters(self.layer_names)):
                experiment.config.set_mse_threshold(mse)
                experiment.config.set_row_count(self.args.rows)
                experiment.config.set_col_count(self.args.cols)
                experiment.config.set_look_back_parameter(self.args.cols)

    def create_experiment_from_name(self, config: CGPConfiguration):
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.args, self.dtype)
        result = parse("mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name:", name)
        
        mse = int(result["mse"])
        rows = int(result["rows"])
        cols = int(result["cols"])
        experiment.config.set_mse_threshold(mse)
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        experiment.add_filter_selectors(self._get_filters(self.layer_names))
        experiment._planner.finish_mapping()
        return experiment

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
        parser.add_argument("--rows", type=int, default=30, help="Number of rows per filter")
        parser.add_argument("--cols", type=int, default=7, help="Number of columns per layer")
        MultiExperiment.get_argument_parser(parser)
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return AllLayersExperiment(config, model_adapter, cgp, args,
                                   layer_names=args.layer_names,
                                   prefix=args.prefix,
                                   suffix=args.suffix,
                                   mse_thresholds=args.mse_thresholds,
                                   batches=args.batches)
