import argparse
from typing import List, Dict, Union, Optional
import torch
import os
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment, FilterSelector
from models.adapters.model_adapter import ModelAdapter
from pathlib import Path

class SkipExperimentError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class MultiExperiment(Experiment):
    def __init__(self, config: Union[str, CGPConfiguration, Path], model_adapter: ModelAdapter, cgp: CGP, args,  dtype=torch.int8) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype)
        self.experiments: Dict[str, Experiment] = {}
        self.args = args

    def _clone(self, config: CGPConfiguration):
        raise NotImplementedError()

    def create_experiment(self, experiment_name: str, filters: Union[List[FilterSelector], FilterSelector], register: bool=True) -> Experiment:    
        config = self.config.clone(self.base_folder / experiment_name / self.config.path.name) \
            if isinstance(self.config, CGPConfiguration) else \
            CGPConfiguration(self.base_folder / experiment_name / Experiment.train_cgp_name)

        new_experiment = Experiment(config, self._model_adapter, self._cgp, self.args, self.dtype)
        new_experiment.parent = self
        filters = filters if isinstance(filters, list) else [filters]

        for x in filters[:-1]:
            new_experiment.add_filter_selector(x)
            new_experiment.next_input_combination()
        new_experiment.add_filter_selector(filters[-1])

        if register:
            self.experiments[new_experiment.get_name()] = new_experiment
        return new_experiment

    def register_experiment(self, experiment: Experiment):
        self.experiments[experiment.get_name()] = experiment

    def get_experiments(self):
        for experiment_name in os.listdir(self.base_folder):
            try:
                config = CGPConfiguration(self.base_folder / experiment_name / Experiment.train_cgp_name)
                new_experiment = Experiment(config, self._model_adapter, self._cgp, self.args, self.dtype)
                new_experiment.parent = self
                self.experiments[experiment_name] = new_experiment
                yield new_experiment
            except FileNotFoundError as e:
                print(f"warn: {str(e)}")       

    def get_experiment(self, experiment_name: str):
        return self.experiments.get(experiment_name)

    def get_number_of_experiments(self) -> int:
        return sum([1 for obj in os.listdir(self.experiment_root_path) if os.path.isfile(self.experiment_root_path / obj)])

    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        return parser
