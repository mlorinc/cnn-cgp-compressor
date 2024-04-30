from typing import Optional, Self, Union
from models.datasets.imagenet import ImageNetKaggle
from models.adapters.model_adapter import ModelAdapter
from torchvision.models.mobilenetv2 import InvertedResidual
import torch.nn as nn
import torch.optim as optim
# import torchvision.models.quantization as models
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.models.quantization as quantization_models

from typing import Any, Optional, Union

class Residual(object):
    def __init__(self, inverted_residual: InvertedResidual) -> None:
        self.dw: nn.Conv2d = inverted_residual.conv[0][0]
        self.pw: nn.Conv2d = inverted_residual.conv[1]

class ExpandedResidual(object):
    def __init__(self, inverted_residual: InvertedResidual) -> None:
        self.expander: nn.Conv2d = inverted_residual.conv[0][0]
        self.dw: nn.Conv2d = inverted_residual.conv[1][0]
        self.pw: nn.Conv2d = inverted_residual.conv[2]
        
class MobileNetBlock(object):
    def __init__(self, block: Union[Residual, ExpandedResidual, nn.Conv2d]) -> None:
        self.block = block

    # def get_parameters_size()

class MobileNetV2Adapter(ModelAdapter):
    name = "mobilenet_v2"
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 1, 2],
        [6, 24, 1, 2],
        [6, 32, 1, 2],
        [6, 32, 1, 2],
        [6, 32, 1, 2],
        [6, 64, 1, 2],
        [6, 64, 1, 2],
        [6, 64, 1, 2],
        [6, 64, 1, 2],
        [6, 96, 1, 1],
        [6, 96, 1, 1],
        [6, 96, 1, 1],
        [6, 160, 1, 2],
        [6, 160, 1, 2],
        [6, 160, 1, 2],
        [6, 320, 1, 1],
    ]    
    
    def __init__(self):
        # super().__init__(models.mobilenet_v2(weights=models.MobileNet_V2_QuantizedWeights, quantize=True, pretrained=True))
        # super().__init__(qunatization_models.mobilenet_v2(weights=qunatization_models.MobileNet_V2_QuantizedWeights, quantize=True, backend = "fbgemm"))
        super().__init__(quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights, pretrained=True, quantize=True))
        self.block_count = 19

    def clone(self):
        return super().clone()

    def get_block(self, index: int) -> Union[Residual, ExpandedResidual, nn.Conv2d]:
        if index == 0 or len(self.inverted_residual_setting) + 1 <= index:
            return self.model.features[index][0]
        
        t, _, _, _ = self.inverted_residual_setting[index-1]
        if t != 1:
            return ExpandedResidual(self.model.features[index - 1])
        else:
            return Residual(self.model.features[index - 1])

    def load(self, path: str = None, inline: bool | None = True) -> Self:
        raise NotImplementedError()

    def get_train_data(self):
        return datasets.ImageNet(root="C:/Users/Majo/imagenet", train=True, download=True, transform=models.MobileNet_V2_Weights.transforms)

    def get_test_data(self):
        return datasets.ImageNet(root="C:/Users/Majo/imagenet", train=False, download=True, transform=models.MobileNet_V2_Weights.transforms)
    
    # def get_test_data(self, **kwargs):
    #     return ImagenetTestDataset("C:/Users/Majo/imagenet/test", models.MobileNet_V2_QuantizedWeights.transforms)

    def get_criterion(self, **kwargs):
        return None

    def get_optimizer(self, **kwargs) -> optim.Optimizer:
        raise NotImplementedError()

def init(model_path: Optional[str]) -> MobileNetV2Adapter:
    return MobileNetV2Adapter()
