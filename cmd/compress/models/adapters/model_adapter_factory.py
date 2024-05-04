from typing import Optional

from models.adapters.model_adapter import ModelAdapter
from models.adapters.mobilenet_adapter import MobileNetV2Adapter
from models.adapters.base import BaseAdapter


def create_adapter(name: str, model_path: Optional[str]) -> ModelAdapter:
    if name != MobileNetV2Adapter.name:
        return BaseAdapter.load_base_model(name, model_path)
    else:
        adapter = MobileNetV2Adapter()
        if model_path:
            adapter.load(model_path)        
        return adapter
