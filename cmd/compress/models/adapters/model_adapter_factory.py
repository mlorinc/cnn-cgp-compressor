from typing import Optional

from models.adapters.model_adapter import ModelAdapter
from models.adapters.mobilenet_adapter import MobileNetV2Adapter
from models.adapters.base import BaseAdapter


def create_adapter(name: str, model_path: Optional[str]) -> ModelAdapter:
    """
    Create a model adapter based on the given model name.

    This function returns a specific adapter for the MobileNetV2 model or a base adapter for other models.

    Args:
        name (str): The name of the model.
        model_path (Optional[str]): The file path to the model's state dictionary.

    Returns:
        ModelAdapter: The appropriate model adapter based on the model name.
    """    
    if name != MobileNetV2Adapter.name:
        return BaseAdapter.load_base_model(name, model_path)
    else:
        adapter = MobileNetV2Adapter()
        if model_path:
            adapter.load(model_path)        
        return adapter
