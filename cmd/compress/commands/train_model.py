import importlib
from models.base_model import BaseModel
from typing import Optional

def get_model(model_name: str, model_path: Optional[str] = None) -> BaseModel:
    return importlib.import_module(f"models.{model_name}").init(model_path)

def train_model(model_name: str, model_path: str, base: Optional[str] = None):
    print(f"Training model: {model_name}")
    model: BaseModel = get_model(model_name, model_path)
    if base:
        model.load(base)
        print(f"Loaded from {base}")

    for train_loss, val_loss in model.fit(yield_on_improve=True):
        model.save()
        print(f"Saved the best model with train loss {train_loss:.6f} and validation loss: {val_loss:.6f}")
    
    model.save(model_path)
    return model
