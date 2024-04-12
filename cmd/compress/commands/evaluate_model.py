import importlib
from commands.train_model import get_model
from models.adapters.model_adapter import ModelAdapter
from typing import Optional

def get_model_adapter(model_name: str, model_path: Optional[str] = None) -> ModelAdapter:
    return importlib.import_module(f"models.adapters.{model_name}").init_adapter(model_path)

def evaluate_model(model_name: str, model_path: str):
    print(f"Evaluating model: {model_name}")
    model: ModelAdapter = get_model_adapter(model_name, model_path)
    model.load(model_path)
    acc, loss = model.evaluate(max_batches=None)
    print(f"acc: {acc:.12f}%, loss {loss:.12f}")
    return acc, loss
