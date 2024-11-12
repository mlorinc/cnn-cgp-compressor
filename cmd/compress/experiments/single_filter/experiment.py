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
# experiment.py: Experiment with single kernel.

import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
import argparse

class SingleFilterExperiment(Experiment):
    """
    SingleFilterExperiment conducts experiments on a single filter of a convolutional neural network layer.
    """    
    name = "single_filter"
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                args,
                grid_size=(5, 5),
                layer_name="conv1",
                channel=0,
                filter=0,
                patience=100000,
                prefix="",
                suffix="",
                dtype=torch.int8) -> None:
        """
        Initialize the SingleFilterExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            args: Additional arguments for the experiment.
            grid_size (tuple, optional): Grid size for the CGP. Defaults to (5, 5).
            layer_name (str, optional): Name of the layer. Defaults to "conv1".
            channel (int, optional): Channel index. Defaults to 0.
            filter (int, optional): Filter index. Defaults to 0.
            patience (int, optional): Number of generations to run before stopping. Defaults to 100000.
            prefix (str, optional): Prefix for the experiment name. Defaults to "".
            suffix (str, optional): Suffix for the experiment name. Defaults to "".
            dtype (torch.dtype, optional): Data type for the experiment. Defaults to torch.int8.
        """        
        super().__init__(config, model_adapter, cgp, args, dtype)
        self.set_paths(self.base_folder / (prefix + self.base_folder.name + suffix))
        self.grid_size = grid_size
        self.layer_name = layer_name

        self.add_filter_selector(self._get_filter(layer_name, filter_i=filter, channel_i=channel))
        self.config.set_patience(patience)
        self.config.set_row_count(grid_size[0])
        self.config.set_col_count(grid_size[1])
        self.config.set_look_back_parameter(grid_size[1])

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
    
    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("--layer-name", default="conv1", help="Name of the CNN layer")
        parser.add_argument("--channel", type=int, default=0, help="Channel index")
        parser.add_argument("--filter", type=int, default=0, help="Filter index")
        parser.add_argument("--grid-size", nargs=2, type=int, default=[5, 5], help="Grid size (rows, columns)")
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return SingleFilterExperiment(config, model_adapter, cgp, args,
                                        grid_size=args.grid_size,
                                        layer_name=args.layer_name,
                                        channel=args.channel,
                                        filter=args.filter,
                                        patience=args.patience,
                                        suffix=args.suffix,
                                        prefix=args.prefix)
