from pathlib import Path
import os

class Datastore(object):
    def __init__(self) -> None:
        self.path = Path(os.environ.get("datastore"))
    def derive(self, path: str) -> Path:
        return self.path / path
    def derive_from_experiment(self, experiment):
        return self.path / experiment.get_name()
    def init_experiment_path(self, experiment):
        self.figures(experiment).mkdir(exist_ok=True, parents=True)
        self.data(experiment).mkdir(exist_ok=True, parents=True)
    def figures(self, experiment):
        return self.derive_from_experiment(experiment) / "figures"
    def models(self, model_name: str):
        return self.derive("models") / model_name
    def data(self, experiment):
        return self.derive_from_experiment(experiment)
