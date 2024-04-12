from models.adapters.base import BaseAdapter
from models.qat_quantized_lenet import QATQuantizedLeNet5, init
from typing import Optional

class QATQuantizedLeNet5Adapter(BaseAdapter):
    def __init__(self, model: QATQuantizedLeNet5) -> None:
        super().__init__(model, True)

def init_adapter(model_path: Optional[str]) -> QATQuantizedLeNet5Adapter:
    return QATQuantizedLeNet5Adapter(init(model_path))
