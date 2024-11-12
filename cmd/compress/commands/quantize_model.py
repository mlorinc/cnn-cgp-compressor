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
# quantize_model.py: Load unquantized model and quantize it. Save results to state dict file.

from commands.train_model import get_model
from models.quantized_model import QuantizedBaseModel

def quantize_model(model_name, model_path, new_path):
    """
    Quantizes a given model and saves the quantized version to a new path.

    Args:
        model_name (str): The name of the model to be quantized.
        model_path (str): The path to the unquantized model.
        new_path (str): The path where the quantized model will be saved.

    Returns:
        QuantizedBaseModel: The quantized model.

    Workflow:
        1. Retrieves the model using `get_model` function.
        2. Ensures the model is an instance of `QuantizedBaseModel`.
        3. Loads the unquantized model from the specified path.
        4. Quantizes the model and saves it to the new path.
    """    
    print(f"Quantizing model: {model_name} and saving as {new_path}")
    model = get_model(model_name, model_path)
    assert isinstance(model, QuantizedBaseModel)
    model.load_unquantized(model_path)
    model.quantize(new_path)
    model.save(new_path)
    return model
