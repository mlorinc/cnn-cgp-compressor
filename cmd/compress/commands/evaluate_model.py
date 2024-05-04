from commands.train_model import get_model
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from models.adapters.model_adapter_factory import create_adapter
from typing import Optional

def get_model_adapter(model_name: str, model_path: Optional[str] = None) -> ModelAdapter:
    return BaseAdapter.load_base_model(model_name, model_path)


def evaluate_base_model(model_name: str, model_path: str, weights: str, args):
    print(f"Evaluating model: {model_name}")
    model: ModelAdapter = create_adapter(model_name, model_path)
    acc, loss = model.evaluate(max_batches=None, **vars(args))
    print(acc, loss)
    return acc, loss
