from typing import Optional
from models.lenet import LeNet5
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from typing import Self

class QuantLeNet5(LeNet5):
    name = "quant_lenet"
    def __init__(self, model_path: str = None):
        super(QuantLeNet5, self).__init__(model_path)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _on_improve(self, average_validation_loss: float):
        pass

    def forward(self, x):
        x = self.quant(x)
        x = super(QuantLeNet5, self).forward(x)
        x = self.dequant(x)
        return x
    
    def prepare(self) -> Self:
        # refer to https://pytorch.org/docs/master/quantization.html#quantization-aware-training
        self.eval()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        self_fp32_fused = torch.ao.quantization.fuse_modules(self,
            [["conv1", "relu1"], ["conv2", "relu2"]])
        
        self_f32_prepared = torch.ao.quantization.prepare_qat(self_fp32_fused.train())
        return self_f32_prepared

    def finish(self, self_f32_prepared: nn.Module) -> Self:
        self_f32_prepared.eval()
        model_int8 = torch.ao.quantization.convert(self_f32_prepared)
        return model_int8

def init(model_path: Optional[str]) -> LeNet5:
    return QuantLeNet5(model_path)
