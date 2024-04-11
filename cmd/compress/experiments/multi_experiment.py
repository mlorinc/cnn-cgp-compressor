from typing import List, Dict, Union, Optional
import torch
import os
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment, FilterSelector
from models.base import BaseModel

class MultiExperiment(Experiment):
    def __init__(self, config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8) -> None:
        super().__init__(config, model, cgp, dtype)
        self.experiments: Dict[str, Experiment] = {}

    def _clone(self, config: CGPConfiguration):
        raise NotImplementedError()

    def create_experiment(self, experiment_name: str, filters: Union[List[FilterSelector], FilterSelector]) -> Experiment:
        new_experiment = Experiment(self.config.clone(self.base_folder / experiment_name / self.config.path.name), self._model, self._cgp, self.dtype)
        new_experiment.parent = self
        filters = filters if isinstance(filters, list) else [filters]

        for x in filters[:-1]:
            new_experiment.add_filter_selector(x)
            new_experiment.next_input_combination()
        new_experiment.add_filter_selector(filters[-1])

        self.experiments[experiment_name] = new_experiment
        return new_experiment

    def get_experiment(self, experiment_name: str):
        return self.experiments.get(experiment_name)

    def get_number_of_experiments(self) -> int:
        return sum([1 for obj in os.listdir(self.experiment_root_path) if os.path.isfile(self.experiment_root_path / obj)])
