from models.adapters.model_adapter import ModelAdapter
from models.base_model import BaseModel
from models.lenet import LeNet5
from models.qat_quantized_lenet import QATQuantizedLeNet5
from models.ptq_quantized_lenet import PTQQuantizedLeNet5
from models.mobilenet_v2 import MobileNetV2
from typing import Optional, Self

class BaseAdapter(ModelAdapter):
    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)
    
    def get_test_data(self, **kwargs):
        assert isinstance(self.model, BaseModel)
        return self.model.get_test_data(**kwargs)
    
    def get_criterion(self, **kwargs):
        assert isinstance(self.model, BaseModel)
        return self.model.get_criterion(**kwargs)

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
        adapter = BaseAdapter.from_base_model(name, path)
        adapter.load()
        return adapter

