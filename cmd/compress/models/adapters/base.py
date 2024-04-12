from models.adapters.model_adapter import ModelAdapter
from models.base_model import BaseModel
from typing import Optional, Self

class BaseAdapter(ModelAdapter):
    def __init__(self, model: BaseModel, quantized: bool) -> None:
        super().__init__(model, quantized)
    
    def evaluate(self, batch_size: int = 32, max_batches: int = None, top: int = 1):
        return self.model.evaluate(batch_size, max_batches, top)

    def load(self, path: str, quantized: Optional[bool] = None, inline: Optional[bool] = False) -> Self:
        quantized = quantized if quantized is not None else self.quantized
        model = self.model.load(path, quantized)
        if not inline:
            myself = self.clone()
            myself.model = model
            myself.quantized = quantized
            return myself
        self.model = model
        self.quantized = quantized
        return self
