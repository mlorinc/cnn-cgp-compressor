from models.adapters.base import BaseAdapter
from models.lenet import LeNet5, init
from typing import Optional

class LeNet5Adapter(BaseAdapter):
    def __init__(self, model: LeNet5) -> None:
        super().__init__(model, False)

def init_adapter(model_path: Optional[str]) -> LeNet5Adapter:
    return LeNet5Adapter(init(model_path))
