import glob
import torch
import pandas as pd
import contextlib
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment
from models.base import BaseModel
from pathlib import Path
from collections import defaultdict

class BaseExperiment(Experiment):
    def __init__(self, experiment_folder: str, experiment_name: str, model: BaseModel, cgp: CGP, dtype=torch.int8, absolute=True) -> None:
        super().__init__(experiment_name, model, cgp, dtype)
        self.experiment_folder_path = Path(experiment_folder)
        self.experiment_folder_path = self.experiment_folder_path.absolute() if absolute else self.experiment_folder_path
        self.experiment_folder_path.mkdir(exist_ok=True, parents=True)
        self.experiment_root_path = self.experiment_folder_path / experiment_name
        self.configs = self.experiment_root_path / "cgp_configs"
        self.weights = self.experiment_root_path / "weights"
        self.initial_acc = None
        self.initial_loss = None

    def _get_cgp_evaluate_file(self, run=None) -> str:
        return self.configs / ("cgp.{run}.config" if run is None else f"cgp.{run}.config")
    
    def _get_cgp_train_file(self, run=None) -> str:
        return self.experiment_root_path / "train_cgp.config"

    def _get_weight_output_file(self, run=None) -> str:
        return self.weights / ("weights.{run}.txt" if run is None else f"weights.{run}.txt")
    
    def _get_train_weight_file(self, run=None) -> str:
        return self.experiment_root_path / "train.data"

    def _get_train_statistics_file(self, run=None) -> str:
        return self.experiment_root_path / "train_statistics.csv"

    def _get_evaluate_statistics_file(self, run=None) -> str:
        return self.experiment_root_path / "evaluate_statistics.csv"

    def _get_stdout_file(self, run=None) -> str:
        return self.experiment_root_path / "stdout.txt"
    
    def _get_stderr_file(self, run=None) -> str:
        return self.experiment_root_path / "stderr.txt"

    def get_number_of_experiment_results(self) -> int:
        return len(glob.glob(f"{str(self._get_cgp_output_file())}.*"))

    def _before_train(self, config: CGPConfiguration):
        config = super()._before_train(config)
        self.configs.mkdir(exist_ok=False, parents=True)
        self.weights.mkdir(exist_ok=False, parents=True)
        return config

    def _recover_empty_experiment(self, config: CGPConfiguration):
        config = super()._recover_empty_experiment(config)
        self.configs.mkdir(exist_ok=True, parents=True)
        self.weights.mkdir(exist_ok=True, parents=True)
        return config

    def evaluate_runs(self):
        runs = range(self.get_number_of_experiment_results())
        if self.initial_acc is None or self.initial_loss is None:
            self.initial_acc, self.initial_loss = self._model.evaluate()
        accuracies = []
        losses = []
        sources = []
        acc_delta = []
        loss_delta = []

        for run in runs:
            try:
                after_acc, after_loss = self.get_model_statistics_from_file(run=run)
            except FileNotFoundError as e:
                if run is not None:
                    self.evaluate(run=run)
                    after_acc, after_loss = self.get_model_statistics_from_file(run=run)
                else:
                    raise e
            accuracies.append(after_acc)
            losses.append(after_loss)
            sources.append(run)
            acc_delta.append(self.initial_acc - after_acc)
            loss_delta.append(self.initial_loss - after_loss)

        data = {"sources": sources, "accuracy": accuracies, "loss": losses, "accuracy_change": acc_delta, "loss_change": loss_delta}
        df = pd.DataFrame(data)
        df_model = pd.DataFrame({"accuracy": [self.initial_acc], "loss": [self.initial_loss]})
        print(df)
        df.to_csv(self.experiment_root_path / "evaluation_stats.csv", index=False)
        df_model.to_csv(self.experiment_root_path / "model_stats.csv", index=False)
        return df