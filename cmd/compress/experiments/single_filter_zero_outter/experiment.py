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
# experiment.py: Experiment to test how algorithm reacts when all output weights are expected to be 0.

import argparse
import torch
import operator
from functools import partial, reduce
import random
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.grid_size.experiment import GridSizeExperiment
from models.selector import FilterSelector
from models.quantization import conv2d_selector, quantize_per_tensor

class SingleFilterZeroOutterExperiment(GridSizeExperiment):
    """
    SingleFilterZeroOutterExperiment performs experiments on single filters of a CNN layer, focusing on 
    zeroing out the outer regions of the filters.
    """    
    name = "single_filter_zero_outter"
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP,
                 dtype=torch.int8, 
                 automatic_creation: bool = True,
                 **kwargs
                 ) -> None:
        """
        Initialize the SingleFilterZeroOutterExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype, optional): Data type for the experiment. Defaults to torch.int8.
            automatic_creation (bool, optional): Flag for automatic creation of experiments. Defaults to True.
            kwargs: Additional arguments for the experiment.
        """        
        super().__init__(config, model_adapter, cgp, automatic_creation=automatic_creation, dtype=dtype, **kwargs)

    def _prepare_filters(self):
        """
        Prepare filters for the experiments, zeroing out the outer regions of the filters.
        """        
        zero_mse_experiments = []
        other_mse_experiments = []

        if self._model_adapter is not None:
            for layer_name in self.layer_names:
                layer = self._model_adapter.get_layer(layer_name)
                for i in range(layer.out_channels):
                    for j in range(layer.in_channels):
                        sel = self._get_filter(layer_name, i, j)
                        self.zero_outter(sel)
                        has_zero = self.has_zero_in_input(sel)
                        zero_suffix = "_zero" if has_zero else ""
                        for experiment in self.create_experiment(f"{self.prefix}{layer_name}_{i}_{j}{zero_suffix}{self.suffix}", sel, register=False):                       
                            if has_zero:
                                experiment.config.set_gate_count_early_stop(0)
                                zero_mse_experiments.append(experiment)
                            else:
                                experiment.config.set_gate_count_early_stop(1)
                                other_mse_experiments.append(experiment)
                            
        selected_other_experiments = random.sample(other_mse_experiments, k=len(zero_mse_experiments))
        
        for a, b in zip(selected_other_experiments, zero_mse_experiments):
            self.register_experiment(a)
            self.register_experiment(b)

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        """
        Get the filter selector for the specified layer, filter index, and channel index.

        Args:
            layer_name (str): Name of the layer.
            filter_i (int): Filter index.
            channel_i (int): Channel index.

        Returns:
            FilterSelector: The filter selector.
        """        
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)

    def zero_outter(self, sel: FilterSelector):
        """
        Zero out the outer regions of the selected filter.

        Args:
            sel (FilterSelector): The filter selector.
        """        
        bias = self._model_adapter.get_bias(sel.selector)
        fp32_weights = self._model_adapter.get_weights(sel.selector)
        for output_selector in sel.out:
            w = fp32_weights[*output_selector]
            size = reduce(operator.mul, w.shape)
            fp32_weights[*output_selector] = quantize_per_tensor(torch.zeros(size), w.q_scale(), w.q_zero_point())
        self._model_adapter.set_weights_bias(sel.selector, fp32_weights, bias)        

    def has_zero_in_input(self, sel: FilterSelector):
        """
        Check if there are zeros in the input of the selected filter.

        Args:
            sel (FilterSelector): The filter selector.

        Returns:
            bool: True if there are zeros in the input, False otherwise.
        """        
        fp32_weights = self._model_adapter.get_train_weights(sel.selector)
        for input_sel in sel.inp:
            w: torch.Tensor = fp32_weights[*input_sel]
            if torch.any(w == 0):
                return True
        return False

    @classmethod
    def with_cli_arguments(cls, config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        """
        Create an instance of SingleFilterZeroOutterExperiment using CLI arguments.

        Args:
            cls: The class to create an instance of.
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            args: Additional arguments from CLI.

        Returns:
            SingleFilterZeroOutterExperiment: The created instance.
        """        
        cls = cls if not args.reuse else partial(SingleFilterZeroOutterExperiment.with_replication, args.reuse, args.name_format)
        return cls(config, model_adapter, cgp, **args)
