import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.composite.experiment import MultiExperiment
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
from models.selector import FilterSelector, FilterSelectorCombination, FilterSelectorCombinations
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
                 dtype=torch.int8, 
                 layer_names=["conv1", "conv2"], 
                 mse_thresholds=thresholds,
                 kernel_dimension=5,
                 kernel_core_dimension=3,
                 prefix="",
                 suffix="",
                 prepare=True,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.layer_names = layer_names
        self.kernel_dimension = kernel_dimension
        self.kernel_core_dimension = kernel_core_dimension
        
        if prepare:
            self._prepare_filters()

    def _prepare_filters(self):
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}mse_{mse}_{self.args['rows']}_{self.args['cols']}{self.suffix}", self._get_filters(self.layer_names)):
                experiment.config.set_output_count(max(map(self._get_output_count, self.layer_names)))
                experiment.config.set_mse_threshold(int(mse**2 * experiment.config.get_output_count()))
                experiment.config.set_dataset_size(len(self.layer_names))
                experiment.config.set_row_count(self.args["rows"])
                experiment.config.set_col_count(self.args["cols"])
                experiment.config.set_look_back_parameter(self.args["cols"])

    def _get_output_count(self, layer) -> int:
        layer: torch.nn.Conv2d = self._model_adapter.get_layer(layer)
        return layer.in_channels * layer.out_channels * (self.kernel_dimension ** 2 - self.kernel_core_dimension ** 2)

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
        for layer_name in layer_names:
            combination = FilterSelectorCombination()
            combination.add(conv2d_selector(layer_name, [slice(None), slice(None)], self.kernel_dimension, self.kernel_core_dimension))
            combinations.add(combination)
        return combinations
