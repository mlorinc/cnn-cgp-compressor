from typing import Optional

from models.adapters.model_adapter import ModelAdapter
from models.adapters.mobilenet_adapter import MobileNetV2Adapter
from models.adapters.base import BaseAdapter


def create_adapter(name: str, model_path: Optional[str]) -> ModelAdapter:
    try:
        adapter = BaseAdapter.load_base_model(name, model_path)
        return adapter
    except ValueError:
        if name == MobileNetV2Adapter.name:
            return MobileNetV2Adapter()    
