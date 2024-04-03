import glob
import torch
import pandas as pd
import contextlib
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment
from models.base import BaseModel
from pathlib import Path

class BaseExperiment(Experiment):
    def __init__(self, experiment_folder: str, experiment_name: str, model: BaseModel, cgp: CGP, dtype=torch.int8) -> None:
        super().__init__(experiment_name, model, cgp, dtype)
        self.experiment_folder_path = Path(experiment_folder)
        self.experiment_folder_path.mkdir(exist_ok=True, parents=True)
        self.experiment_root_path = self.experiment_folder_path / experiment_name
        self.configs = self.experiment_root_path / "cgp_configs"
        self.weights = self.experiment_root_path / "weights"

    def _get_cgp_output_file(self) -> str:
        return self.configs / "data.cgp"
    
    def _get_weight_output_file(self) -> str:
        return self.weights / "inferred_weights"
    
    def _get_train_file(self) -> str:
        return self.experiment_root_path / "train.data"

    def _get_statistics_file(self) -> str:
        return self.experiment_root_path / "statistics.csv"

    def _get_stdout_file(self) -> str:
        return self.experiment_root_path / "stdout.txt"
    
    def _get_stderr_file(self) -> str:
        return self.experiment_root_path / "stderr.txt"

    def get_number_of_experiment_results(self) -> int:
        return len(glob.glob(f"{str(self._get_cgp_output_file())}.*"))

    def _before_train(self, config: CGPConfiguration):
        super()._before_train(config)
        self.configs.mkdir(exist_ok=False, parents=True)
        self.weights.mkdir(exist_ok=False, parents=True)

    def _recover_empty_experiment(self, config: CGPConfiguration):
        config = super()._recover_empty_experiment(config)
        self.configs.mkdir(exist_ok=True, parents=True)
        self.weights.mkdir(exist_ok=True, parents=True)
        return config
