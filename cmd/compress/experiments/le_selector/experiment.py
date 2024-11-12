# Copyright 2024 Marián Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# experiment.py: Experiment testing CGP algorithm capabilities on LeNet-5 model with objective to create circuit
# that would infer all weights without any input weights. It has only 1 input set to zero as workaround because
# of CGP algorithm requirements to have at least 1 input. That weight is constatly set to 0 not to infere with result.

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
    """
    A class to manage LeSelector experiments, extending the MultiExperiment class.
    This experiment is extreme case and was named from initial idea that every layer
    would be approximated from no weights. Initially, multiplexed layers were planned,
    however due to technical complications all layers are approxiamted at once.
    """
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
        """
        Initialize the LeSelectorExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to torch.int8.
            layer_names (list, optional): List of layer names. Defaults to ["conv1", "conv2"].
            mse_thresholds (list, optional): List of MSE thresholds. Defaults to [0].
            prefix (str, optional): Prefix for experiment names. Defaults to "".
            suffix (str, optional): Suffix for experiment names. Defaults to "".
            prepare (bool, optional): Whether to prepare filters automatically. Defaults to True.
            **kwargs: Additional arguments.
        """        
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.layer_names = layer_names
        
        if prepare:
            self._prepare_filters()

    def _prepare_filters(self):
        """
        Prepare filters for the experiments.
        """        
        output_count = 16*6*25+6*25
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}mse_{mse}_{self.args['rows']}_{self.args['cols']}{self.suffix}", self._get_filters(self.layer_names)):
                experiment.config.set_mse_threshold(self.error_threshold_function(output_count, error=mse))
                experiment.config.set_row_count(self.args["rows"])
                experiment.config.set_col_count(self.args["cols"])
                experiment.config.set_output_count(output_count)
                experiment.config.set_look_back_parameter(int(self.args["cols"] + 1))

    def create_experiment_from_name(self, config: CGPConfiguration):
        """
        Create an experiment instance from the configuration name.

        Args:
            config (CGPConfiguration): Configuration for the CGP.

        Returns:
            Experiment: The created experiment instance.
        """        
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.args, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error, **self.args)
        result = parse("mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name:", name)
        
        mse = float(result["mse"])
        rows = int(result["rows"])
        cols = int(result["cols"])
        output_count = 16*6*25+6*25
        experiment.config.set_mse_threshold(self.error_threshold_function(output_count, error=mse))
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        experiment.set_feature_maps_combinations(self._get_filters(self.layer_names))
        return experiment

    def _get_filters(self, layer_names: List[str]):
        """
        Get filter selectors for the specified layers.

        Args:
            layer_names (List[str]): List of layer names.

        Returns:
            FilterSelectorCombinations: The filter selector combinations.
        """        
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
