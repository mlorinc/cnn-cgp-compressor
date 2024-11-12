# Copyright 2024 Marián Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# model_adapter_factory.py: Adapter factory to build adapter instances from name and optionally load them from given path.

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
