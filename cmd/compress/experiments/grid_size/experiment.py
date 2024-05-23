import argparse
import torch
import random
from pathlib import Path
from typing import Dict, Generator, List, Self, Tuple, Union
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from models.quantization import conv2d_selector
from experiments.composite.experiment import MultiExperiment
from functools import partial

class GridSizeExperiment(MultiExperiment):
    """
    A class to manage grid size experiments, extending the MultiExperiment class.
    Grid size experiment aims to test multiple grid size configurations to find
    optimal grid size,
    """    
    name = "grid_size"
    default_grid_sizes=[(5, 5)]
    def __init__(self, 
                config: CGPConfiguration,
                model_adapter: ModelAdapter, 
                cgp: CGP,
                grid_sizes=default_grid_sizes,
                layer_names=["conv1", "conv2"],
                prefix="",
                suffix="",
                dtype=torch.int8,
                n: int = 1,
                automatic_creation: bool = True,
                **kwargs
                ) -> None:
        """
        Initialize the GridSizeExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            grid_sizes (list, optional): List of grid sizes. Defaults to [(5, 5)].
            layer_names (list, optional): List of layer names. Defaults to ["conv1", "conv2"].
            prefix (str, optional): Prefix for experiment names. Defaults to "".
            suffix (str, optional): Suffix for experiment names. Defaults to "".
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to torch.int8.
            n (int, optional): Number of samples. Defaults to 1.
            automatic_creation (bool, optional): Whether to automatically create experiments. Defaults to True.
            **kwargs: Additional arguments.
        """        
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)    
        assert n is None or n % len(layer_names) == 0
        self.grid_sizes = grid_sizes
        self.layer_names = layer_names
        self.prefix = prefix
        self.suffix = suffix
        self.k = n // len(layer_names) if n is not None else None
        self._prepare_filters(automatic_creation)
        
    def _prepare_filters(self, automatic_creation: bool):
        """
        Prepare filters for the experiments.

        Args:
            automatic_creation (bool): Whether to automatically create experiments.
        """        
        if automatic_creation:
            for layer_name in self.layer_names:
                layer = self._model_adapter.get_layer(layer_name)
                combinations = [(a, b) for b in range(layer.out_channels) for a in range(layer.in_channels)]
                combinations = random.sample(combinations, k=self.k) if self.k is not None else combinations
                for row, col in self.grid_sizes:
                    for inp, out in combinations:
                        for experiment in self.setup_experiment(layer_name, inp, out, row, col):
                            self.register_experiment(experiment)
            
    def _get_filter(self, layer_name: str, filter_i: int, channel_i: int):
        """
        Get the filter selector for a specific layer.

        Args:
            layer_name (str): Name of the layer.
            filter_i (int): Filter index.
            channel_i (int): Channel index.

        Returns:
            FilterSelector: The filter selector.
        """        
        return conv2d_selector(layer_name, [filter_i, channel_i], 5, 3)

    def setup_experiment(self, layer_name: str, inp: int, out: int, row: int, col: int, register: bool = False) -> Generator[Self, None, None]:
        """
        Set up an experiment.

        Args:
            layer_name (str): Name of the layer.
            inp (int): Input index.
            out (int): Output index.
            row (int): Number of rows.
            col (int): Number of columns.
            register (bool, optional): Whether to register the experiment. Defaults to False.

        Yields:
            Generator[Self, None, None]: The set up experiment.
        """        
        layer_name = str(layer_name); inp = int(inp); out = int(out)
        row = int(row); col = int(col)
        
        for experiment in self.create_experiment(f"{self.prefix}{layer_name}_x_{out}_{inp}_{row}_{col}{self.suffix}", self._get_filter(layer_name, out, inp), register=register, name_fmt=self.name_fmt):
            experiment.config.set_row_count(row)
            experiment.config.set_col_count(col)
            experiment.config.set_look_back_parameter(col)
            yield experiment

    @classmethod
    def with_replication(clf, reference_path: Union[Path, str], name_fmt: str, path: Union[Path, str], model_adapter: ModelAdapter, cgp: CGP, args, **kwargs):
        """
        Create an instance of GridSizeExperiment with replication.

        Args:
            reference_path (Union[Path, str]): Path to the reference experiment.
            name_fmt (str): Format string for naming experiments.
            path (Union[Path, str]): Path to the new experiment.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            GridSizeExperiment: The created instance.
        """        
        instance = MultiExperiment(reference_path, model_adapter, cgp, args, name_fmt=name_fmt)
        desired_instance = clf(path, model_adapter, cgp, args, automatic_creation=False, **kwargs)
        for experiment in instance.get_experiments():
            data = experiment.decode_configuration()
            for _ in desired_instance.setup_experiment(data["layer_name"], data["inp"], data["out"], data["row"], data["col"], register=True):
                pass
        desired_instance._prepared = True
        return desired_instance

    @classmethod
    def with_cli_arguments(cli, config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        """
        Create an instance of GridSizeExperiment with CLI arguments.

        Args:
            cli (type): The class type.
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            args: Additional arguments.

        Returns:
            GridSizeExperiment: The created instance.
        """        
        cls = cli if not args.reuse else partial(GridSizeExperiment.with_replication, args.reuse, args.name_format)
        return cls(config, model_adapter, cgp, **args)
