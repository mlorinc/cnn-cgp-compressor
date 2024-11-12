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
# debug_model.py: Helper file to enable model debugging in IDE. Set breakpoint to return statement before use.

from models.adapters.model_adapter import ModelAdapter
from models.adapters.model_adapter_factory import create_adapter

def debug_model(model_name: str, model_path: str):
    print(f"Evaluating model: {model_name}")
    model: ModelAdapter = create_adapter(model_name, model_path)
    return
