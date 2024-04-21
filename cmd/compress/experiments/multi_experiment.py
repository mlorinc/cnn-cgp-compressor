import argparse
from typing import Generator, List, Dict, Union, Optional, Self
import torch
import os
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment, FilterSelector
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from pathlib import Path
from glob import glob
from abc import ABC, abstractmethod

class SkipExperimentError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class MultiExperiment(Experiment, ABC):
    def __init__(self, config: Union[str, CGPConfiguration, Path], model_adapter: ModelAdapter, cgp: CGP, args, dtype=torch.int8, name_fmt: str = None, batches: int = None) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype)
        self.experiments: Dict[str, Experiment] = {}
        self.args = args
        self.name_fmt = name_fmt
        self.batches = batches

    @classmethod
    def with_data_only(cls, path: Union[Path, str], model_name: str = None, model_path: str = None, cgp: str = None) -> Self:
        path = Path(path)
        model_adapter = BaseAdapter.load_base_model(model_name, model_path) if model_name and model_path else None
        return cls(path, model_adapter, cgp, {}, prepare=False)     

    def _clone(self, config: CGPConfiguration):
        raise NotImplementedError()

    def __contains__(self, experiment: Experiment):
        if isinstance(experiment, Experiment):
            return experiment.get_name() in self.experiments
        elif isinstance(experiment, str):
            return experiment in self.experiments
        else:
            raise TypeError("invalid type for in operator")

    @abstractmethod
    def create_experiment_from_name(self, config: CGPConfiguration) -> Experiment:
        raise NotImplementedError()

    def create_experiment(self, experiment_name: str, filters: Union[List[FilterSelector], FilterSelector], register: bool=True, name_fmt: str = None) -> Generator[Experiment, None, None]:   
        for i in range(self.batches or 1):
            current_experiment_name = experiment_name
            
            if self.batches is not None:
                current_experiment_name = experiment_name + f"_batch_{i}"

            config = self.config.clone(self.base_folder / current_experiment_name / self.config.path.name) \
                if isinstance(self.config, CGPConfiguration) else \
                CGPConfiguration(self.base_folder / current_experiment_name / Experiment.train_cgp_name)

            new_experiment = Experiment(config, self._model_adapter, self._cgp, self.args, self.dtype)
            new_experiment.parent = self
            
            if self.batches is not None:
                errors = []
                if new_experiment.config.has_start_generation():
                    errors.append("it is not allowed to set start generation when using batches")
                if new_experiment.config.has_start_generation():
                    errors.append("it is not allowed to set start run when using batches")
                if new_experiment.config.get_number_of_runs() % self.batches != 0:
                    errors.append(f"number of runs must be divisible by number of batches: batches={self.batches}, runs={new_experiment.config.get_number_of_runs()}")
                if errors:
                    raise ValueError("the following errors occured while batching:\n\t" + "\n\t".join(errors))
                batch_step = new_experiment.config.get_number_of_runs() // self.batches
                new_experiment.config.set_start_run(batch_step * i)
                new_experiment.config.set_number_of_runs(batch_step * i + batch_step)
            new_experiment.add_filter_selectors(filters)
            if register:
                self.register_experiment(new_experiment)
            new_experiment.name_fmt = name_fmt + "_batch_{i}" if name_fmt else None
            yield new_experiment

    def register_experiment(self, experiment: Experiment):
        self.experiments[experiment.get_name()] = experiment

    def remove_experiment(self, experiment: Experiment):
        del self.experiments[experiment.get_name()]

    def get_experiments(self):
        for experiment_name in os.listdir(self.base_folder):
            try:
                config = CGPConfiguration(self.base_folder / experiment_name / Experiment.train_cgp_name)
                new_experiment = self.create_experiment_from_name(config)
                new_experiment.name_fmt = self.name_fmt
                new_experiment.parent = self
                self.experiments[experiment_name] = new_experiment
                yield new_experiment
            except FileNotFoundError as e:
                print(f"warn: {str(e)}")       
    
    def get_experiments_with_glob(self, str_glob: str):
        for experiment_name in glob(str(self.base_folder / str_glob)):
            print(experiment_name)
            pass
            try:
                config = CGPConfiguration(self.base_folder / experiment_name / Experiment.train_cgp_name)
                new_experiment = self.create_experiment_from_name(config)
                new_experiment.parent = self
                self.experiments[experiment_name] = new_experiment
                yield new_experiment
            except FileNotFoundError as e:
                print(f"warn: {str(e)}")      

    def get_experiment(self, experiment_name: str, from_filesystem: bool = False):
        if not from_filesystem:
            return self.experiments.get(experiment_name)
        else:
            for _ in self.get_experiments(): pass
            return self.experiments.get(experiment_name) 

    def get_number_of_experiments(self) -> int:
        return sum([1 for obj in os.listdir(self.experiment_root_path) if os.path.isfile(self.experiment_root_path / obj)])

    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("-b", "--batches", default=None, type=int, help="Split single experiment into multiple smaller batches")    
        return parser
