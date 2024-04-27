from typing import Generator, Self
import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.composite.experiment import MultiExperiment
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
import experiments.single_channel.cli as sc
from parse import parse

class SingleChannelExperiment(MultiExperiment):
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
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.dtype)
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
        experiment.add_filter_selectors(self._get_filter(result["layer_name"], 0))
        experiment._planner.finish_mapping()
        return experiment
            
    def _get_filter(self, layer_name: str, channel_i: int):
        return conv2d_selector(layer_name, [slice(None), channel_i], 5, 3)
