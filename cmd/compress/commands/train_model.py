import importlib
from models.base_model import BaseModel
from typing import Optional

def get_model(model_name: str, model_path: Optional[str] = None) -> BaseModel:
    """
    Retrieves a model instance by importing its module and initializing it.

    Args:
        model_name (str): The name of the model to import.
        model_path (Optional[str]): The optional path to the model.

    Returns:
        BaseModel: An instance of the specified model.
    """    
    return importlib.import_module(f"models.{model_name}").init(model_path)

def train_model(model_name: str, model_path: str, base: Optional[str] = None):
    """
    Trains a specified model and saves the best version based on training and validation losses.

    Args:
        model_name (str): The name of the model to train.
        model_path (str): The path where the trained model will be saved.
        base (Optional[str]): An optional path to a base model to load before training.

    Returns:
        BaseModel: The trained model.

    Workflow:
        1. Retrieves the model using `get_model` function.
        2. Loads the base model if provided.
        3. Trains the model, yielding on improvement of loss.
        4. Saves the model with the best training and validation losses.
    """    
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
