import glob
from typing import Generator, Tuple
import torch
import contextlib
import os
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.simple_experiment import BaseExperiment
from models.base import BaseModel

class MultiExperiment(BaseExperiment):
    def __init__(self, experiment_folder: str, experiment_name: str, model: BaseModel, cgp: CGP, dtype=torch.int8) -> None:
        super().__init__(experiment_folder, experiment_name, model, cgp, dtype)

    def _set_experiment_name(self, experiment_name: str):
        self.experiment_root_path = self.experiment_folder_path / experiment_name
        self.configs = self.experiment_root_path / "cgp_configs"
        self.weights = self.experiment_root_path / "weights"

    @contextlib.contextmanager
    def experiment_context(self, experiment_name: str) -> Generator[Tuple[str, CGPConfiguration], None, None]:
        old_name = self.experiment_name
        try:
            self.reset()
            self._set_experiment_name(os.path.join(old_name, experiment_name))
            yield self.experiment_root_path.name, self._cgp.config.clone()
        finally:
            self.reset()
            self._set_experiment_name(old_name)

    def get_number_of_experiment_results(self) -> int:
        return len(glob.glob(f"{str(self._get_cgp_output_file())}.*"))

    def get_number_of_experiments(self) -> int:
        return sum([1 for obj in os.listdir(self.experiment_root_path) if os.path.isfile(self.experiment_root_path / obj)])
