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
    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)
    
    def get_convolution_layers(self) -> Iterable[Conv2d]:
        raise NotImplementedError()
    
    def get_test_data(self, **kwargs):
        assert isinstance(self.model, BaseModel)
        return self.model.get_test_data(**kwargs)
    
    def get_train_data(self, **kwargs):
        pass  
    
    def get_criterion(self, **kwargs):
        assert isinstance(self.model, BaseModel)
        return self.model.get_criterion(**kwargs)

    def get_custom_dataset(self, train_dataset: bool = False, **kwargs):
        if train_dataset:
            return self.model.get_train_data(**kwargs)
        else:
            return self.get_test_data(**kwargs)

    def clone(self):
        assert isinstance(self.model, BaseModel)
        model = self.model._create_self()
        model.load()
        return BaseAdapter(model)

    def load(self, path: str = None, inline: Optional[bool] = True) -> Self:
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
        if "lenet" not in name:
            adapter = BaseAdapter.from_base_model(name, path)
        else:
            adapter = LeNet5Adapater.from_base_model(name, path)
        adapter.load()
        return adapter

class LeNet5Adapater(BaseAdapter):
    def __init__(self, model: LeNet5) -> None:
        super().__init__(model)
    def get_convolution_layers(self):
        return [self.model.conv1, self.model.conv2]
