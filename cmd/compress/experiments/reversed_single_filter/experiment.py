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
# experiment.py: Test how algorithm can infer weights from outter borders to core weights.

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
    """
    A class for conducting experiments with reversed single filters, extending GridSizeExperiment.
    Similarly, grid sizes are examining however outer weights are used for inference.
    """    
    name = "reversed_single_filter"
    default_grids = [(2, 2), (3, 3), (5, 5), (10, 10)]
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                automatic_creation: bool = True,
                **kwargs) -> None:
        """
        Initialize the ReversedSingleFilterExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            automatic_creation (bool, optional): Whether to automatically create experiments. Defaults to True.
            **kwargs: Additional arguments.
        """        
        super().__init__(config, model_adapter, cgp, automatic_creation=automatic_creation, **kwargs)         

    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        """
        Get the filter selector with reversed input and output channels.

        Args:
            layer_name (str): Name of the layer.
            filter_i (int): Filter index.
            channel_i (int): Channel index.

        Returns:
            FilterSelector: The filter selector with reversed input and output channels.
        """        
        sel = super()._get_filter(layer_name, filter_i, channel_i)
        sel.inp, sel.out = sel.out, sel.inp
        return sel
    
    @classmethod
    def with_cli_arguments(cls, config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        """
        Create an instance of ReversedSingleFilterExperiment using command-line arguments.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            args: Command-line arguments.

        Returns:
            ReversedSingleFilterExperiment: The created instance.
        """        
        cls = cls if not args.reuse else partial(ReversedSingleFilterExperiment.with_replication, args.reuse, args.name_format)
        return cls(config, model_adapter, cgp, **args)