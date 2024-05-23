from typing import Generator, List, Dict, Union, Optional, Self
import torch
import os
from experiments.composite.cli import get_argument_parser
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment
from models.selector import FilterSelector, FilterSelectorCombinations
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from pathlib import Path
from glob import glob
from abc import ABC, abstractmethod

class SkipExperimentError(ValueError):
    """
    Custom exception raised when an experiment should be skipped.
    """    
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class MultiExperiment(Experiment, ABC):
    """
    A class to manage multiple experiments, extending the base Experiment class.
    """    
    def __init__(self, config: Union[str, CGPConfiguration, Path], model_adapter: ModelAdapter, cgp: CGP, dtype=torch.int8, name_fmt: str = None, batches: int = None, **kwargs) -> None:
        """
        Initialize the MultiExperiment class.

        Args:
            config (Union[str, CGPConfiguration, Path]): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype): Data type for the tensors.
            name_fmt (str, optional): Format string for naming experiments.
            batches (int, optional): Number of batches for experiments.
            **kwargs: Additional arguments.
        """        
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        self.experiments: Dict[str, Experiment] = {}
        self.name_fmt = name_fmt
        self.batches = batches

    @classmethod
    def with_data_only(cls, path: Union[Path, str], model_name: str = None, model_path: str = None, cgp: str = None, e_fitness="SE") -> Self:
        """
        Create a MultiExperiment instance with only data.

        Args:
            path (Union[Path, str]): Path to the data.
            model_name (str, optional): Name of the model.
            model_path (str, optional): Path to the model.
            cgp (str, optional): CGP instance.
            e_fitness (str, optional): Fitness evaluation method.

        Returns:
            Self: An instance of MultiExperiment.
        """        
        path = Path(path)
        model_adapter = BaseAdapter.load_base_model(model_name, model_path) if model_name and model_path else None
        return cls(path, model_adapter, cgp, {}, prepare=False, e_fitness=e_fitness)     

    def _clone(self, config: CGPConfiguration):
        """
        Clone the experiment configuration.

        Args:
            config (CGPConfiguration): The configuration to clone.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """        
        raise NotImplementedError()

    def __contains__(self, experiment: Experiment):
        """
        Check if an experiment is contained within this MultiExperiment.

        Args:
            experiment (Experiment or str): The experiment to check.

        Returns:
            bool: True if the experiment is contained, False otherwise.
        """        
        if isinstance(experiment, Experiment):
            return experiment.get_name() in self.experiments
        elif isinstance(experiment, str):
            return experiment in self.experiments
        else:
            raise TypeError("invalid type for in operator")

    @abstractmethod
    def create_experiment_from_name(self, config: CGPConfiguration) -> Experiment:
        """
        Create an experiment from its name.

        Args:
            config (CGPConfiguration): Configuration for the CGP.

        Returns:
            Experiment: The created experiment.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """        
        raise NotImplementedError()

    def create_experiment(self, experiment_name: str, feature_maps_combiations: FilterSelectorCombinations, register: bool=True, name_fmt: str = None) -> Generator[Experiment, None, None]:   
        """
        Create and register a new experiments.

        Args:
            experiment_name (str): The name of the experiment.
            feature_maps_combiations (FilterSelectorCombinations): Filter combinations for the experiment.
            register (bool, optional): Whether to register the experiment.
            name_fmt (str, optional): Format string for naming the experiment.

        Yields:
            Generator[Experiment, None, None]: The created experiment.
        """        
        base_experiment = None
        for i in range(self.batches or 1):
            current_experiment_name = experiment_name
            
            if self.batches is not None:
                current_experiment_name = experiment_name + f"_batch_{i}"
            config = self.config.clone(self.base_folder / current_experiment_name / self.config.path.name) \
                if isinstance(self.config, CGPConfiguration) else \
                CGPConfiguration(self.base_folder / current_experiment_name / Experiment.train_cgp_name)
            
            new_experiment = Experiment(config, self._model_adapter, self._cgp, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error, start_run=self._start_run, **self.args)
            new_experiment.parent = self
            
            if self.batches is not None:
                errors = []
                base_experiment = base_experiment or new_experiment
                if new_experiment.config.has_start_generation():
                    errors.append("it is not allowed to set start generation when using batches")
                if new_experiment.config.has_start_generation():
                    errors.append("it is not allowed to set start run when using batches")
                if new_experiment.config.get_number_of_runs() % self.batches != 0:
                    errors.append(f"number of runs must be divisible by number of batches: batches={self.batches}, runs={new_experiment.config.get_number_of_runs()}")
                if errors:
                    raise ValueError("the following errors occured while batching:\n\t" + "\n\t".join(errors))
                batch_step = new_experiment.config.get_number_of_runs() // self.batches
                new_experiment._start_run = batch_step * i
                new_experiment.config.set_start_run(new_experiment._start_run)
                new_experiment.config.set_number_of_runs(new_experiment._start_run + batch_step)
            
            new_experiment.batched_parent = base_experiment if i > 0 else None
            new_experiment.set_feature_maps_combinations(feature_maps_combiations)
            if register:
                self.register_experiment(new_experiment)
            new_experiment.name_fmt = name_fmt + "_batch_{i}" if name_fmt else None
            yield new_experiment

    def register_experiment(self, experiment: Experiment):
        """
        Register an experiment.

        Args:
            experiment (Experiment): The experiment to register.
        """        
        self.experiments[experiment.get_name()] = experiment

    def remove_experiment(self, experiment: Experiment):
        """
        Remove an experiment from the registry.

        Args:
            experiment (Experiment): The experiment to remove.
        """        
        del self.experiments[experiment.get_name()]

    def get_experiments(self):
        """
        Get all registered experiments.

        Yields:
            Generator[Experiment, None, None]: The registered experiments.
        """        
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
    
    def get_experiments_with_glob(self, str_glob: str, return_names=False):
        """
        Get experiments matching a glob pattern.

        Args:
            str_glob (str): The glob pattern to match.
            return_names (bool, optional): Whether to return experiment names.

        Yields:
            Union[Experiment, str]: The matching experiments or their names.
        """        
        for experiment_name in glob(str(self.base_folder / str_glob)):
            path = Path(experiment_name)
            experiment_name = path.name
            print(experiment_name)
            try:
                config = CGPConfiguration(self.base_folder / experiment_name / Experiment.train_cgp_name)
                new_experiment = self.create_experiment_from_name(config)
                new_experiment.parent = self
                self.experiments[experiment_name] = new_experiment
                if not return_names:
                    yield new_experiment
                else:
                    yield experiment_name
            except FileNotFoundError as e:
                print(f"warn: {str(e)}")      

    def get_experiment(self, experiment_name: str, from_filesystem: bool = False):
        """
        Get a specific experiment by name.

        Args:
            experiment_name (str): The name of the experiment.
            from_filesystem (bool, optional): Whether to load the experiment from the filesystem.

        Returns:
            Experiment: The requested experiment.
        """        
        if not from_filesystem:
            return self.experiments.get(experiment_name)
        else:
            for _ in self.get_experiments(): pass
            return self.experiments.get(experiment_name) 

    def get_number_of_experiments(self) -> int:
        """
        Get the number of registered experiments.

        Returns:
            int: The number of registered experiments.
        """        
        return sum([1 for obj in os.listdir(self.experiment_root_path) if os.path.isfile(self.experiment_root_path / obj)])
