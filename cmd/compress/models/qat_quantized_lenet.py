from typing import Optional
from models.lenet import LeNet5
import torch.nn as nn
import torch
import copy

class QATQuantizedLeNet5(LeNet5):
    name = "qat_quantized_lenet"

    def __init__(self, model_path: str = None):
        super(QATQuantizedLeNet5, self).__init__(model_path)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.backend = "fbgemm"     

    def _prepare(self):
        # fuse first Conv-ReLU pair
        torch.quantization.fuse_modules(self, ["conv1", "relu1"], inplace=True)
        # fuse second Conv-ReLU pair
        torch.quantization.fuse_modules(self, ["conv2", "relu2"], inplace=True)

        self.train()
        self.qconfig = torch.quantization.get_default_qconfig(self.backend)
        torch.quantization.prepare_qat(self, inplace=True)

    def _convert(self):
        torch.quantization.convert(self, inplace=True)

    def load(self, model_path: str, quantized: bool = True):
            return super().load(model_path, quantized)

    def quantize(self, new_path: str):
        self.model_path = new_path
        reference_model = copy.deepcopy(self)

        self.eval()
        self._prepare()
        self.fit()
        self.eval()
        self._convert()

        # # Sensitity analysis
        # weight_sqnr_dict, activation_sqnr_dict = sensisitivy_analysis(reference_model, self, self._get_test_data())
        
        # print("Weight SQNR Dictionary:", weight_sqnr_dict)
        # print("Activation SQNR Dictionary:", activation_sqnr_dict)

        # 1 byte instead of 4 bytes for FP32
        assert self.conv1.weight().element_size() == 1
        assert self.conv2.weight().element_size() == 1
        assert reference_model.conv1.weight.element_size() == 4

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