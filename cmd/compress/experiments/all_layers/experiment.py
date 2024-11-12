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
# experiment.py: Experiment implementation that infers outter weights using core 3x3 weights on every layer.

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
    """
        All layers experiment focuses on aspect of testing multiplexers by
        approximating kernel core cores to outer kernel values. Each layer
        is split into own dataset hence LeNet-5 has dataset size equal 2.
    """
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
        """
        Initializes the AllLayersExperiment class with the given parameters.
        All layers experiment focuses on aspect of testing multiplexers by
        approximating kernel core cores to outer kernel values. Each layer
        is split into own dataset hence LeNet-5 has dataset size equal 2.

        Args:
            config (CGPConfiguration): Configuration for CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype): Data type for the tensors.
            layer_names (List[str]): List of layer names to be included in the experiment.
            mse_thresholds (List[int]): List of MSE thresholds.
            kernel_dimension (int): The dimension of the kernel.
            kernel_core_dimension (int): The core dimension of the kernel.
            prefix (str): Prefix for experiment names.
            suffix (str): Suffix for experiment names.
            prepare (bool): Flag to indicate whether to prepare the filters.
            **kwargs: Additional arguments.
        """        
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
        """
        Prepares the filters for the experiments by setting up the configurations for each experiment.
        """        
        for mse in self.mse_thresholds:
            input_count = max(map(self._get_input_count, self.layer_names))
            output_count = max(map(self._get_output_count, self.layer_names))
            original_mse = mse
            mse = self.error_threshold_function(output_count + 96, error=mse)
            for experiment in self.create_experiment(f"{self.prefix}mse_{original_mse}_{self.args['rows']}_{self.args['cols']}{self.suffix}", self._get_filters(self.layer_names)):
                experiment.config.set_input_count(input_count)
                experiment.config.set_output_count(output_count)
                experiment.config.set_mse_threshold(mse)
                experiment.config.set_dataset_size(len(self.layer_names))
                experiment.config.set_row_count(self.args["rows"])
                experiment.config.set_col_count(self.args["cols"])
                experiment.config.set_look_back_parameter(int(self.args["cols"]) + 1)

    def _get_output_count(self, layer) -> int:
        """
        Gets the output count for a given layer.

        Args:
            layer (str): The layer name.

        Returns:
            int: The output count for the layer.
        """        
        layer: torch.nn.Conv2d = self._model_adapter.get_layer(layer)
        return layer.in_channels * layer.out_channels * (self.kernel_dimension ** 2 - self.kernel_core_dimension ** 2)

    def _get_input_count(self, layer) -> int:
        """
        Gets the input count for a given layer.

        Args:
            layer (str): The layer name.

        Returns:
            int: The input count for the layer.
        """        
        layer: torch.nn.Conv2d = self._model_adapter.get_layer(layer)
        return layer.in_channels * layer.out_channels * (self.kernel_core_dimension ** 2)

    def create_experiment_from_name(self, config: CGPConfiguration):
        """
        Creates an experiment instance from a given configuration name.

        Args:
            config (CGPConfiguration): The configuration object.

        Returns:
            Experiment: The created experiment instance.
        """        
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error, **self.args)
        result = parse("mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name:", name)
        
        mse = int(float(result["mse"])**2 * (6 * 16 + 6 * 16 * 16))
        rows = int(result["rows"])
        cols = int(result["cols"])
        experiment.config.set_mse_threshold(mse)
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        experiment.set_feature_maps_combinations(self._get_filters(self.layer_names))
        return experiment

    def _get_filters(self, layer_names: List[str]):
        """
        Gets the filters for the specified layer names.

        Args:
            layer_names (List[str]): List of layer names.

        Returns:
            FilterSelectorCombinations: The filter selector combinations for the specified layers.
        """        
        combinations = FilterSelectorCombinations()
        for layer_name in layer_names:
            combination = FilterSelectorCombination()
            combination.add(conv2d_selector(layer_name, [slice(None), slice(None)], self.kernel_dimension, self.kernel_core_dimension))
            combinations.add(combination)
        return combinations
