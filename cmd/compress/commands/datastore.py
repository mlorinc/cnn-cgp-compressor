from pathlib import Path
import os

class Datastore(object):
    """
    A class to manage paths for data storage in experiments.

    Attributes:
        path (Path): The base path for the datastore, derived from the environment variable "datastore".
    """    
    def __init__(self) -> None:
        """
        Initializes the Datastore with the base path from the environment variable "datastore".
        """        
        self.path = Path(os.environ.get("datastore"))
    def derive(self, path: str) -> Path:
        """
        Derives a new path based on the base datastore path.

        Args:
            path (str): The relative path to be appended to the base datastore path.

        Returns:
            Path: The combined path.
        """        
        return self.path / path
    def derive_from_experiment(self, experiment):
        """
        Derives a new path for a given experiment.

        Args:
            experiment: The experiment object which should have a get_name() method.

        Returns:
            Path: The combined path including the experiment name.
        """        
        return self.path / experiment.get_name()
    def init_experiment_path(self, experiment):
        """
        Initializes the necessary directories for an experiment.

        Args:
            experiment: The experiment object which should have a get_name() method.
        """        
        self.figures(experiment).mkdir(exist_ok=True, parents=True)
        self.data(experiment).mkdir(exist_ok=True, parents=True)
    def figures(self, experiment):
        """
        Gets the path to the figures directory for a given experiment.

        Args:
            experiment: The experiment object which should have a get_name() method.

        Returns:
            Path: The path to the figures directory.
        """        
        return self.derive_from_experiment(experiment) / "figures"
    def models(self, model_name: str):
        """
        Gets the path to the models directory for a given model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            Path: The path to the models directory.
        """        
        return self.derive("models") / model_name
    def data(self, experiment):
        """
        Gets the path to the data directory for a given experiment.

        Args:
            experiment: The experiment object which should have a get_name() method.

        Returns:
            Path: The path to the data directory.
        """        
        return self.derive_from_experiment(experiment)
