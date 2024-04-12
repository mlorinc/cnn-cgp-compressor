from models.adapters.base import BaseAdapter
from models.ptq_quantized_lenet import PTQQuantizedLeNet5, init
from typing import Optional

class PTQQuantizedLeNet5Adapter(BaseAdapter):
    def __init__(self, model: PTQQuantizedLeNet5) -> None:
        super().__init__(model, True)

def init_adapter(model_path: Optional[str]) -> PTQQuantizedLeNet5Adapter:
    return PTQQuantizedLeNet5Adapter(init(model_path))
