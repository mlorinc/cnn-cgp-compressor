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
# qat_quantized_lenet.py: Utility model used to quantize LeNet-5 using QAT.

from typing import Optional, Self
from models.lenet import LeNet5
from models.quantized_model import QuantizedBaseModel
import torch
import copy

class QATQuantizedLeNet5(QuantizedBaseModel, LeNet5):
    """
    QATQuantizedLeNet5 class inherits from QuantizedBaseModel and LeNet5.
    This class represents a PyTorch quantized version of the LeNet-5 architecture using Quantization-Aware Training (QAT).

    Attributes:
        name (str): The name of the model, set to "qat_quantized_lenet".
    """    
    name = "qat_quantized_lenet"

    def __init__(self, model_path: str = None):
        super(QATQuantizedLeNet5, self).__init__(model_path)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.backend = "fbgemm"     

    def _create_self(self, *args) -> Self:
        return QATQuantizedLeNet5(self.model_path)

    def _prepare(self):
        """
        Prepares the model for quantization by fusing Conv-ReLU pairs and setting the quantization configuration.
        """        
        super()._prepare()
        # fuse first Conv-ReLU pair
        torch.quantization.fuse_modules(self, ["conv1", "relu1"], inplace=True)
        # fuse second Conv-ReLU pair
        torch.quantization.fuse_modules(self, ["conv2", "relu2"], inplace=True)

        self.train()
        self.qconfig = torch.quantization.get_default_qconfig(self.backend)
        torch.quantization.prepare_qat(self, inplace=True)

    def _convert(self):
        """
        Converts the model to a quantized version.
        """        
        super()._convert()
        torch.quantization.convert(self, inplace=True)

    def quantize(self, new_path: str, inline=True):
        """
        Quantizes the model and saves the quantized model to the specified path.

        Parameters:
            new_path (str): Path to save the quantized model.
            inline (bool): Flag to indicate whether to save the quantized model inline.
        """        
        self.qat_quantization()
        self.save(new_path, inline=inline)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x

def init(model_path: Optional[str]) -> QATQuantizedLeNet5:
    return QATQuantizedLeNet5(model_path)
