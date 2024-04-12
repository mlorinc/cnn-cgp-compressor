from typing import Optional
from models.lenet import LeNet5
import torch
import copy

class PTQQuantizedLeNet5(LeNet5):
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

    def load(self, model_path: str, quantized: bool = True):
        return super().load(model_path, quantized)

    def quantize(self, new_path: str = None):
        self.model_path = new_path or self.model_path
        reference_model = copy.deepcopy(self)
        self.eval()
        self._prepare()

        # Calibrate
        with torch.inference_mode():
            _, val_loader = self._get_train_validation_data()
            for x, _ in val_loader:
                self(x)
    
        self._convert()
        
        # # Sensitity analysis
        # weight_sqnr_dict, activation_sqnr_dict = sensisitivy_analysis(reference_model, self, self._get_test_data())
        
        # print("Weight SQNR Dictionary:", weight_sqnr_dict)
        # print("Activation SQNR Dictionary:", activation_sqnr_dict)

        # 1 byte instead of 4 bytes for FP32
        assert self.conv1.weight().element_size() == 1
        assert self.conv2.weight().element_size() == 1
        assert reference_model.conv1.weight.element_size() == 4

    def _on_improve(self, average_validation_loss: float):
        self.save(self.model_path)
        print(f"Saved the best model with validation loss: {average_validation_loss:.6f}")

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
