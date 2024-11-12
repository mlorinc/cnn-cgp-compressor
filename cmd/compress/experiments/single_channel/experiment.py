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
# experiment.py: Experiment to test inference capabilities on single channel.

from typing import Generator, Self
import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from models.selector import FilterSelectorCombinations, FilterSelectorCombination
from experiments.composite.experiment import MultiExperiment
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
import experiments.single_channel.cli as sc
from parse import parse

class SingleChannelExperiment(MultiExperiment):
    """
    A class for conducting single channel experiments, extending MultiExperiment.
    Approximate a single channel in CNN.
    """    
    name = "single_channel"
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP, 
                 dtype=torch.int8, 
                 layer_name=sc.layer_name, 
                 channel=sc.channel, 
                 prefix="", 
                 suffix="", 
                 mse_thresholds=sc.thresholds,
                 rows_per_filter=sc.rows_per_filter,
                 rows=None,
                 cols=None,
                 prepare=True,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        """
        Initialize the SingleChannelExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype): Data type for the experiment.
            layer_name (str): Name of the layer to be used in the experiment.
            channel (int): Channel to be used in the experiment.
            prefix (str, optional): Prefix for the experiment name. Defaults to "".
            suffix (str, optional): Suffix for the experiment name. Defaults to "".
            mse_thresholds (list, optional): List of MSE thresholds. Defaults to sc.thresholds.
            rows_per_filter (int, optional): Number of rows per filter. Defaults to sc.rows_per_filter.
            rows (int, optional): Number of rows. Defaults to None.
            cols (int, optional): Number of columns. Defaults to None.
            prepare (bool, optional): Whether to prepare filters. Defaults to True.
            **kwargs: Additional arguments.
        """        
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.rows_per_filter = rows_per_filter
        self.layer_name = layer_name
        self.channel = channel
        self.rows = rows
        self.cols = cols
        self._prepare = prepare
        self._prepare_filters()

    def _prepare_filters(self):
        """
        Prepare filters for the experiment if self._prepare is True.
        """        
        if self._prepare:
            layer = self._model_adapter.get_layer(self.layer_name)
            single_cell_size = self.rows_per_filter * layer.out_channels
            rows = self.rows or single_cell_size
            cols = self.cols or 7
            for mse in self.mse_thresholds:
                for experiment in self.create_experiment(f"{self.prefix}{self.layer_name}_mse_{mse}_{rows}_{cols}{self.suffix}", self._get_filter(self.layer_name, self.channel)):
                    experiment.config.set_mse_threshold(mse**2 * experiment.config.get_output_count() * experiment.config.get_dataset_size())
                    experiment.config.set_row_count(rows)
                    experiment.config.set_col_count(cols)
                    experiment.config.set_look_back_parameter(cols)
         
    def create_experiment_from_name(self, config: CGPConfiguration):
        """
        Create an experiment instance from the given configuration.

        Args:
            config (CGPConfiguration): Configuration for the CGP.

        Returns:
            Experiment: The created experiment instance.
        """        
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error, **self.args)
        result = parse("{layer_name}_mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name:", name)
        
        mse = int(result["mse"])
        rows = int(result["rows"])
        cols = int(result["cols"])
        experiment.config.set_mse_threshold(mse**2 * experiment.config.get_output_count() * experiment.config.get_dataset_size())
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        experiment.set_feature_maps_combinations(self._get_filter(result["layer_name"], 0))
        return experiment
            
    def _get_filter(self, layer_name: str, channel_i: int):
        """
        Get the filter selector for the specified layer and channel.

        Args:
            layer_name (str): Name of the layer.
            channel_i (int): Channel index.

        Returns:
            FilterSelectorCombinations: The filter selector combinations.
        """        
        combinations = FilterSelectorCombinations()
        combination = FilterSelectorCombination()
        combinations.add(combination)
        combination.add(conv2d_selector(layer_name, [slice(None), channel_i], 5, 3))
        return combinations
