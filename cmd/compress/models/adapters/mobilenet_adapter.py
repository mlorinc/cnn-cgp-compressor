from typing import Callable, Optional, Self, Union

import torch.quantization.utils
from commands.datastore import Datastore
from models.adapters.model_adapter import ModelAdapter
from torchvision.models.mobilenetv2 import InvertedResidual
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.quantization as quantization_models
import datasets
from torch.utils.data import Dataset
import os

from typing import Optional, Union

class MobileNetDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (Tensor): A tensor containing the data samples.
        """
        self.data = data
        self.transform = quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1.transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            data_point = self.data[idx]
            image, label = data_point["image"], data_point["label"]
            
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            
            if image.shape[0] == 4:
                image = image[:3]
            
            return self.transform(image), label
        except Exception as e:
            print(self.data[idx])
            raise e

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
        super().__init__(quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True))
        self._set_attributes()

    def clone(self):
        state_dict = self.model.state_dict()
        new_instance = MobileNetV2Adapter()
        new_instance.device = self.device
        new_instance.model = quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True)
        new_instance.model.load_state_dict(state_dict)
        return new_instance

    def get_layer(self, selector: Union[str, Callable[[Self], nn.Conv2d]]) -> nn.Module:
        if isinstance(selector, str):
            return getattr(self, selector)
        elif isinstance(selector, nn.Module):
            return selector
        else:
            return selector(self)

    def get_block(self, index: int) -> Union[Residual, ExpandedResidual, nn.Conv2d]:
        if index == 0 or len(self.inverted_residual_setting) < index:
            return self.model.features[index][0]
        
        t, _, _, _ = self.inverted_residual_setting[index-1]
        if t != 1:
            return ExpandedResidual(self.model.features[index])
        else:
            return Residual(self.model.features[index])

    def get_blocks(self):
        for i in range(1, len(self.inverted_residual_setting) + 1):
            yield self.get_block(i) 
    
    def get_convolution_layers(self):
        for i in range(len(self.model.features)):            
            try:
                m = self.get_block(i)
                if i == 0 or i == len(self.model.features) - 1:
                    yield m
                
                if isinstance(m, Residual):
                    yield m.dw
                    yield m.pw
                elif isinstance(m, ExpandedResidual):
                    yield m.expander
                    yield m.dw
                    yield m.pw                                  
                else:
                    print(m)
            except Exception as e:
                print(f"error at index {i}")
                print(self.model.features[i])
                raise e
            
    def _set_attributes(self):
        for i in range(len(self.model.features)):            
            try:
                m = self.get_block(i)
                if i == 0 or i == len(self.model.features) - 1:
                    setattr(self, f"features_{i}_0", m)                
                elif isinstance(m, Residual):
                    setattr(self, f"features_{i}_conv_0_0", m.dw)                
                    setattr(self, f"features_{i}_1_0", m.pw)                
                elif isinstance(m, ExpandedResidual):
                    setattr(self, f"features_{i}_conv_0_0", m.expander)                
                    setattr(self, f"features_{i}_conv_0_1", m.dw)                
                    setattr(self, f"features_{i}_1_0", m.pw)                                                    
                else:
                    print(m)
            except Exception as e:
                print(f"error at index {i}")
                print(self.model.features[i])
                raise e        
    
    def get_all_layers(self):
        for i in range(len(self.model.features)):            
            try:
                m = self.get_block(i)
                if i == 0 or i == len(self.model.features) - 1:
                    yield f"features_{i}_0", m                
                elif isinstance(m, Residual):
                    yield f"features_{i}_conv_0_0", m.dw                
                    yield f"features_{i}_1_0", m.pw               
                elif isinstance(m, ExpandedResidual):
                    yield f"features_{i}_conv_0_0", m.expander                
                    yield f"features_{i}_conv_0_1", m.dw               
                    yield f"features_{i}_conv_1_0", m.pw                                                 
                else:
                    print(m)
            except Exception as e:
                print(f"error at index {i}")
                print(self.model.features[i])
                raise e        
    
    def load(self, path: str = None, inline: bool | None = True) -> Self:
        if inline:
            self.model = quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True)
            self.model.load_state_dict(torch.load(path))
            return self
        else:
            new_adapter = self.clone()
            new_adapter.model.load_state_dict(path)
            return new_adapter

    def _load_dataset(self, split: str, num_proc=1, **kwargs):
        ds = datasets.load_dataset("imagenet-1k",
                                   split=split,
                                   data_dir=Datastore().derive("datasets"),
                                   cache_dir=Datastore().derive(".cache"),
                                   trust_remote_code=True,
                                   streaming=False,
                                   token=os.environ.get("huggingface"),
                                   num_proc=num_proc
                                   ).with_format("torch", device=self.device)        
        return MobileNetDataset(ds)

    def get_train_data(self, **kwargs):
        return self._load_dataset("train", **kwargs)

    def get_validation_data(self, **kwargs):
        return self._load_dataset("validation", **kwargs)

    def get_test_data(self, **kwargs):
        return self._load_dataset("validation", **kwargs)
    
    def get_criterion(self, **kwargs):
        return nn.CrossEntropyLoss()
    
    def get_optimizer(self, **kwargs) -> optim.Optimizer:
        raise NotImplementedError()

def init(model_path: Optional[str]) -> MobileNetV2Adapter:
    return MobileNetV2Adapter()
