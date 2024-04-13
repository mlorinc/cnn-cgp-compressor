from commands.train_model import get_model
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from typing import Optional

def get_model_adapter(model_name: str, model_path: Optional[str] = None) -> ModelAdapter:
    return BaseAdapter.load_base_model(model_name, model_path)

def evaluate_model(model: ModelAdapter, weights: str):
    acc, loss = model.evaluate(max_batches=None)
    print(f"acc: {acc:.12f}%, loss {loss:.12f}")
    return acc, loss

def evaluate_model(model_name: str, model_path: str, weights: str):
    print(f"Evaluating model: {model_name}")
    model: ModelAdapter = BaseAdapter.load_base_model(model_name, model_path)
    acc, loss = model.evaluate(max_batches=None)
    print(f"acc: {acc:.12f}%, loss {loss:.12f}")
    return acc, loss
