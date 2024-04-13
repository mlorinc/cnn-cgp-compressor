from typing import Optional
from models.adapters.base import BaseModel
from models.datasets.imagenet import ImagenetTestDataset
from models.adapters.model_adapter import ModelAdapter
import torch.optim as optim
import torchvision.models.quantization as models

class MobileNetV2Adapter(ModelAdapter):
    name = "mobilenet_v2"
    def __init__(self):
        super().__init__(models.mobilenet_v2(weights=models.MobileNet_V2_QuantizedWeights, quantize=True, pretrained=True))

    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    def get_validation_data(self, **kwargs):
        return super().get_validation_data(**kwargs)

    def get_test_data(self, **kwargs):
        return ImagenetTestDataset("C:/Users/Majo/imagenet/test", models.MobileNet_V2_QuantizedWeights.transforms)

    def get_criterion(self, **kwargs):
        raise NotImplementedError()

    def get_optimizer(self, **kwargs) -> optim.Optimizer:
        raise NotImplementedError()

def init(model_path: Optional[str]) -> MobileNetV2Adapter:
    return MobileNetV2Adapter()
