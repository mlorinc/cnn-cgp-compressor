from typing import Optional
from models.lenet import LeNet5
from models.quantized_model import QuantizedBaseModel
from tqdm import tqdm
import torch
import copy

class PTQQuantizedLeNet5(QuantizedBaseModel, LeNet5):
    name = "ptq_quantized_lenet"

    def __init__(self, model_path: str = None):
        super(PTQQuantizedLeNet5, self).__init__(model_path)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.backend = "fbgemm"     

    def _prepare(self):
        # fuse first Conv-ReLU pair
        torch.quantization.fuse_modules(self, ["conv1", "relu1"], inplace=True)
        # fuse second Conv-ReLU pair
        torch.quantization.fuse_modules(self, ["conv2", "relu2"], inplace=True)
        self.qconfig = torch.quantization.get_default_qconfig(self.backend)
        torch.quantization.prepare(self, inplace=True)

    def _convert(self):
        torch.quantization.convert(self, inplace=True)

    def quantize(self, new_path: str = None, inline=True):
        self.ptq_quantization(new_path)
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

def init(model_path: Optional[str]) -> PTQQuantizedLeNet5:
    return PTQQuantizedLeNet5(model_path)
