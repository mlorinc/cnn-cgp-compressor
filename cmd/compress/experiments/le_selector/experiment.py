import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.composite.experiment import MultiExperiment
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
from models.selector import FilterSelector, ValuesSelector, FilterSelectorCombination, FilterSelectorCombinations
import argparse
from typing import List
from parse import parse

class LeSelectorExperiment(MultiExperiment):
    name = "le_selector"
    thresholds = [0]
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP,
                 dtype=torch.int8, 
                 layer_names=["conv1", "conv2"], 
                 mse_thresholds=thresholds,
                 prefix="",
                 suffix="",
                 prepare=True,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.layer_names = layer_names
        
        if prepare:
            self._prepare_filters()

    def _prepare_filters(self):
        output_count = 16*6*25+6*25
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}mse_{mse}_{self.args['rows']}_{self.args['cols']}{self.suffix}", self._get_filters(self.layer_names)):
                experiment.config.set_mse_threshold(self.error_threshold_function(output_count, error=mse))
                experiment.config.set_row_count(self.args["rows"])
                experiment.config.set_col_count(self.args["cols"])
                experiment.config.set_output_count(output_count)
                experiment.config.set_look_back_parameter(int(self.args["cols"] + 1))

    def create_experiment_from_name(self, config: CGPConfiguration):
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.args, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error)
        result = parse("{prefix}mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name:", name)
        
        mse = int(result["mse"])
        rows = int(result["rows"])
        cols = int(result["cols"])
        experiment.config.set_mse_threshold(mse)
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        experiment.set_feature_maps_combinations(self._get_filters(self.layer_names))
        return experiment

    def _get_filters(self, layer_names: List[str]):
        combinations = FilterSelectorCombinations()
        combination = FilterSelectorCombination()
        for layer_name in layer_names:
            combination.add(FilterSelector(layer_name, [ValuesSelector([0])], [(slice(None), slice(None), slice(None), slice(None))]))
        combinations.add(combination)
        return combinations

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return LeSelectorExperiment(config, model_adapter, cgp, args,
                                   layer_names=args.layer_names,
                                   prefix=args.prefix,
                                   suffix=args.suffix,
                                   mse_thresholds=args.mse_thresholds,
                                   batches=args.batches)
