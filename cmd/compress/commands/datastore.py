# Copyright 2024 Marián Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# datastore.py: Data managment class for various experiment data.

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
