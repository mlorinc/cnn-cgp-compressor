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
# base.py: Adapter to handle various models and provide single API.

from pathlib import Path

from torch.nn.modules import Conv2d

from commands.datastore import Datastore
from models.adapters.model_adapter import ModelAdapter
from models.base_model import BaseModel
from models.lenet import LeNet5
from models.qat_quantized_lenet import QATQuantizedLeNet5
from models.ptq_quantized_lenet import PTQQuantizedLeNet5
from typing import Iterable, Optional, Self
from abc import ABC, abstractmethod

class BaseAdapter(ModelAdapter, ABC):
    """
    A base adapter class for handling different types of neural network models.

    Attributes:
        model (BaseModel): The neural network model.

    Methods:
        get_convolution_layers(): Abstract method to get convolution layers of the model.
        get_test_data(**kwargs): Returns the test data for the model.
        get_train_data(**kwargs): Placeholder for getting training data.
        get_criterion(**kwargs): Returns the criterion used by the model.
        get_custom_dataset(train_dataset=False, **kwargs): Returns a custom dataset.
        clone(): Clones the current adapter.
        load(path=None, inline=True): Loads the model from the specified path.
    """    
    def __init__(self, model: BaseModel) -> None:
        """
        Initializes the BaseAdapter with the given model.

        Args:
            model (BaseModel): The neural network model.
        """        
        super().__init__(model)
    
    def get_convolution_layers(self) -> Iterable[Conv2d]:
        """
        Abstract method to get convolution layers of the model.

        Returns:
            Iterable[Conv2d]: An iterable of convolution layers.
        """        
        raise NotImplementedError()
    
    def get_test_data(self, **kwargs):
        """
        Returns the test data for the model.

        Args:
            **kwargs: Additional arguments for getting test data.

        Returns:
            Dataset: The test dataset.
        """        
        assert isinstance(self.model, BaseModel)
        return self.model.get_test_data(**kwargs)
    
    def get_train_data(self, **kwargs):
        """
        Placeholder for getting training data.

        Args:
            **kwargs: Additional arguments for getting training data.
        """        
        pass  
    
    def get_criterion(self, **kwargs):
        """
        Returns the criterion used by the model.

        Args:
            **kwargs: Additional arguments for getting the criterion.

        Returns:
            nn.Module: The criterion used by the model.
        """        
        assert isinstance(self.model, BaseModel)
        return self.model.get_criterion(**kwargs)

    def get_custom_dataset(self, train_dataset: bool = False, **kwargs):
        """
        Returns a custom dataset.

        Args:
            train_dataset (bool): Flag to indicate if training dataset is required.
            **kwargs: Additional arguments for getting the custom dataset.

        Returns:
            Dataset: The custom dataset.
        """        
        if train_dataset:
            return self.model.get_train_data(**kwargs)
        else:
            return self.get_test_data(**kwargs)

    def clone(self):
        """
        Clones the current adapter.

        Returns:
            BaseAdapter: A cloned instance of the current adapter.
        """        
        assert isinstance(self.model, BaseModel)
        model = self.model._create_self()
        model.load()
        return BaseAdapter(model)

    def load(self, path: str = None, inline: Optional[bool] = True) -> Self:
        """
        Loads the model from the specified path.

        Args:
            path (str, optional): The path to load the model from.
            inline (Optional[bool], optional): Flag to indicate if the model should be loaded inline.

        Returns:
            Self: The loaded adapter.
        """        
        model = self.model if inline else self.model.clone()
        assert path or model.model_path is not None
        model.load(path or model.model_path)
        if not inline:
            myself = self.clone()
            myself.model = model
            return myself
        self.model = model
        return self

    @classmethod
    def from_base_model(cls, name: str, path: str) -> Self:
        """
        Creates an adapter from a base model.

        Args:
            name (str): The name of the model.
            path (str): The path to the model.

        Returns:
            Self: The created adapter.
        """        
        datastore = Datastore()
        
        if path:
            util_path = Path(path)
            if not util_path.exists() and datastore.derive(f"models/{path}").exists():
                path = str(datastore.derive(f"models/{path}").absolute())
        else:
            raise ValueError("path must not be none")
        
        if name == LeNet5.name:
            return cls(LeNet5(path))
        if name == QATQuantizedLeNet5.name:
            return cls(QATQuantizedLeNet5(path))
        if name == PTQQuantizedLeNet5.name:
            return cls(PTQQuantizedLeNet5(path))    
        else:
            raise ValueError(f"unknown model {name}")                

    @staticmethod
    def load_base_model(name: str, path: str) -> Self:
        """
        Loads a base model.

        Args:
            name (str): The name of the model.
            path (str): The path to the model.

        Returns:
            Self: The loaded adapter.
        """        
        if "lenet" not in name:
            adapter = BaseAdapter.from_base_model(name, path)
        else:
            adapter = LeNet5Adapater.from_base_model(name, path)
        adapter.load()
        return adapter

class LeNet5Adapater(BaseAdapter):
    """
    Adapter class for LeNet5 model.

    Methods:
        get_convolution_layers(): Returns the convolution layers of the model.
    """    
    def __init__(self, model: LeNet5) -> None:
        """
        Initializes the LeNet5Adapter with the given model.

        Args:
            model (LeNet5): The LeNet5 model.
        """        
        super().__init__(model)
    def get_convolution_layers(self):
        """
        Returns the convolution layers of the model.

        Returns:
            list: A list of convolution layers.
        """        
        return [self.model.conv1, self.model.conv2]
