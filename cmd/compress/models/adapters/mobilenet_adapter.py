from typing import Callable, Optional, Self, Union
from functools import reduce
import operator
import os

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
from parse import parse
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
    
    expected_weight_count = {
        "features.0.0": 864,
        "features.1.conv.0.0": 288,
        "features.1.conv.1": 512,
        "features.2.conv.0.0": 1536,
        "features.2.conv.1.0": 864,
        "features.2.conv.2": 2304,
        "features.3.conv.0.0": 3456,
        "features.3.conv.1.0": 1296,
        "features.3.conv.2": 3456,
        "features.4.conv.0.0": 3456,
        "features.4.conv.1.0": 1296,
        "features.4.conv.2": 4608,
        "features.5.conv.0.0": 6144,
        "features.5.conv.1.0": 1728,
        "features.5.conv.2": 6144,
        "features.6.conv.0.0": 6144,
        "features.6.conv.1.0": 1728,
        "features.6.conv.2": 6144,
        "features.7.conv.0.0": 6144,
        "features.7.conv.1.0": 1728,
        "features.7.conv.2": 12288,
        "features.8.conv.0.0": 24576,
        "features.8.conv.1.0": 3456,
        "features.8.conv.2": 24576,
        "features.9.conv.0.0": 24576,
        "features.9.conv.1.0": 3456,
        "features.9.conv.2": 24576,
        "features.10.conv.0.0": 24576,
        "features.10.conv.1.0": 3456,
        "features.10.conv.2": 24576,
        "features.11.conv.0.0": 24576,
        "features.11.conv.1.0": 3456,
        "features.11.conv.2": 36864,
        "features.12.conv.0.0": 55296,
        "features.12.conv.1.0": 5184,
        "features.12.conv.2": 55296,
        "features.13.conv.0.0": 55296,
        "features.13.conv.1.0": 5184,
        "features.13.conv.2": 55296,
        "features.14.conv.0.0": 55296,
        "features.14.conv.1.0": 5184,
        "features.14.conv.2": 92160,
        "features.15.conv.0.0": 153600,
        "features.15.conv.1.0": 8640,
        "features.15.conv.2": 153600,
        "features.16.conv.0.0": 153600,
        "features.16.conv.1.0": 8640,
        "features.16.conv.2": 153600,
        "features.17.conv.0.0": 153600,
        "features.17.conv.1.0": 8640,
        "features.17.conv.2": 307200,
        "features.18.0": 409600,
        "classifier.1": 1280000
    }
    
    def __init__(self):
        # super().__init__(models.mobilenet_v2(weights=models.MobileNet_V2_QuantizedWeights, quantize=True, pretrained=True))
        # super().__init__(qunatization_models.mobilenet_v2(weights=qunatization_models.MobileNet_V2_QuantizedWeights, quantize=True, backend = "fbgemm"))
        super().__init__(quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True))
        self.layers = {}
        self._set_attributes()
        self.model_path = None

    def _new_instance(self):
        new_instance = MobileNetV2Adapter()
        new_instance.device = self.device
        new_instance.model = quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True)
        return new_instance        

    def clone(self):
        if self.model_path is not None:
            return self.load(inline=False)
        else:
            raise NotImplementedError()
            new_adapter = self._new_instance()
            new_adapter.model.state_dict(self.model.state_dict())

    def set_weights(self, layer: str, weights: torch.Tensor):
        if not isinstance(layer, str):
            raise TypeError("only string layer selector is allowed when using Mobilenet")
        
        state_dict = self.model.state_dict()
        state_dict[layer + ".weight"] = weights
        self.model.load_state_dict(state_dict)

    def get_layer(self, selector: Union[str, Callable[[Self], nn.Conv2d]]) -> nn.Module:
        if isinstance(selector, str):
            return self.layers[selector]
        elif isinstance(selector, nn.Module):
            return selector
        else:
            return selector(self)

    def get_weights(self, selector: str):
        if not isinstance(selector, str):
            raise TypeError("only string layer selector is allowed when using Mobilenet")
        return self.model.state_dict()[selector + ".weight"]

    def get_train_weights(self, selector: str):    
        return self.get_weights(selector).int_repr()

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
    
    def _to_implementation_name(self, layer: str) -> str:
        new_name = None
        result = parse("features_{i:d}_0", layer)     
        if not new_name and result is not None:
            new_name = "features.{i:d}.0".format(**result.named)     
        result = parse("features_{i:d}_conv_0_0", layer)     
        if not new_name and result is not None:
            new_name = "features.{i:d}.conv.0.0".format(**result.named)        
        result = parse("features_{i:d}_conv_0_1", layer)
        if not new_name and result is not None:
            new_name = "features.{i:d}.conv.1.0".format(**result.named)          
        result = parse("features_{i:d}_1_0", layer)
        if not new_name and result is not None:
            i = int(result["i"])
            t, _, _, _ = self.inverted_residual_setting[i-1]
            if t != 1:
                new_name = "features.{i:d}.conv.2".format(**result.named)
            else:
                new_name = "features.{i:d}.conv.1".format(**result.named)
                
        result = parse("features.{i:d}.conv.{n:d}", new_name or layer)
        result = result or parse("features.{i:d}.conv.{n:d}", new_name or layer)
        if result:
            i, n = int(result["i"]), int(result["n"])
            t, _, _, _ = self.inverted_residual_setting[i-1]
            expected_n = 2 if t != 1 else 1
            
            if expected_n != n:
                raise ValueError(f"invalid layer name {new_name or layer}; expected the last filter to have index {expected_n}, got instead {n}")          
        
        if new_name and self.expected_weight_count[new_name] != reduce(operator.mul, self.get_weights(new_name).shape):
            raise ValueError(f"invalid number of weights for {new_name}; expeted: {self.expected_weight_count[new_name]}, got: {reduce(operator.mul, self.get_weights(new_name).shape)}")
        
        if parse("features.{i:d}.0", layer)\
            or parse("features.{i:d}.conv.0.0", layer)\
            or parse("features.{i:d}.conv.1.0", layer)\
            or parse("features.{i:d}.conv.1", layer)\
            or parse("features.{i:d}.conv.2", layer):
            new_name = layer        
        
        if new_name:
            return new_name
        
        raise ValueError(f"unknown layer {layer}")     
    
    def _set_attributes(self):
        for i in range(len(self.model.features)):            
            try:
                m = self.get_block(i)
                if i == 0 or i == len(self.model.features) - 1:
                    self.layers[f"features_{i}_0"] = m
                    self.layers[f"features.{i}.0"] = m
                    assert self.expected_weight_count[f"features.{i}.0"] == reduce(operator.mul, m.weight().shape)
                elif isinstance(m, Residual):
                    self.layers[f"features_{i}_conv_0_0"] = m.dw
                    self.layers[f"features_{i}_1_0"] = m.pw
                    self.layers[f"features.{i}.conv.0.0"] = m.dw
                    self.layers[f"features.{i}.conv.1"] = m.pw
                    assert self.expected_weight_count[f"features.{i}.conv.0.0"] == reduce(operator.mul, m.dw.weight().shape)
                    assert self.expected_weight_count[f"features.{i}.conv.1"] == reduce(operator.mul, m.pw.weight().shape)
                elif isinstance(m, ExpandedResidual):
                    self.layers[f"features_{i}_conv_0_0"] = m.expander
                    self.layers[f"features_{i}_conv_0_1"] = m.dw
                    self.layers[f"features_{i}_1_0"] = m.pw
                    self.layers[f"features.{i}.conv.0.0"] = m.expander
                    self.layers[f"features.{i}.conv.1.0"] = m.dw
                    self.layers[f"features.{i}.conv.2"] = m.pw
                    assert self.expected_weight_count[f"features.{i}.conv.0.0"] == reduce(operator.mul, m.expander.weight().shape)
                    assert self.expected_weight_count[f"features.{i}.conv.1.0"] == reduce(operator.mul, m.dw.weight().shape)
                    assert self.expected_weight_count[f"features.{i}.conv.2"] == reduce(operator.mul, m.pw.weight().shape)
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
                    assert self.expected_weight_count[f"features.{i}.0"] == reduce(operator.mul, m.weight().shape)
                    yield f"features.{i}.0", m                
                elif isinstance(m, Residual):
                    assert self.expected_weight_count[f"features.{i}.conv.0.0"] == reduce(operator.mul, m.dw.weight().shape)
                    assert self.expected_weight_count[f"features.{i}.conv.1"] == reduce(operator.mul, m.pw.weight().shape)
                    yield f"features.{i}.conv.0.0", m.dw                
                    yield f"features.{i}.conv.1", m.pw               
                elif isinstance(m, ExpandedResidual):
                    assert self.expected_weight_count[f"features.{i}.conv.0.0"] == reduce(operator.mul, m.expander.weight().shape)
                    assert self.expected_weight_count[f"features.{i}.conv.1.0"] == reduce(operator.mul, m.dw.weight().shape)
                    assert self.expected_weight_count[f"features.{i}.conv.2"] == reduce(operator.mul, m.pw.weight().shape)                                                 
                    yield f"features.{i}.conv.0.0", m.expander                
                    yield f"features.{i}.conv.1.0", m.dw               
                    yield f"features.{i}.conv.2", m.pw
                else:
                    print(m)
            except Exception as e:
                print(f"error at index {i}")
                print(self.model.features[i])
                raise e        
    
    def load(self, path: str = None, inline: bool | None = True) -> Self:
        self.model_path = path or self.model_path
        
        if self.model_path is None:
            raise ValueError("the model does not have path defined")
        
        if inline:
            self.model = quantization_models.mobilenet_v2(weights=quantization_models.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model_path = path
            return self
        else:
            new_adapter = self._new_instance()
            new_adapter.model.load_state_dict(torch.load(self.model_path))
            new_adapter.model_path = self.model_path
            return new_adapter

    def _load_dataset(self, split: str = "validation", num_proc=1, **kwargs):
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

    def get_validation_data(self, split=None, **kwargs):
        return self._load_dataset(split="validation", **kwargs)

    def get_test_data(self, split=None, **kwargs):
        return self._load_dataset(split="validation", **kwargs)
    
    def get_criterion(self, **kwargs):
        return nn.CrossEntropyLoss()
    
    def get_optimizer(self, **kwargs) -> optim.Optimizer:
        raise NotImplementedError()

def init(model_path: Optional[str]) -> MobileNetV2Adapter:
    return MobileNetV2Adapter()
