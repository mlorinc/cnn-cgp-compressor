import torch
import torch.nn as nn
import operator
import copy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from typing import List, Union, Optional, Self
from functools import reduce
from models.quantization import dequantize_per_tensor
from models.selector import FilterSelector

class ModelAdapter(ABC):
    def __init__(self, model: nn.Module, quantized: bool) -> None:
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantized = quantized
        self.model.to(self.device)

    @abstractmethod
    def load(self, path: str, quantized: Optional[bool] = False, inline: Optional[bool] = False) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, batch_size: int = 32, max_batches: int = None, top: int=1):
        raise NotImplementedError()

    def get_layer(self, layer: str) -> nn.Module:
        return getattr(self.model, layer)

    def clone(self):
        cloned_adapter = copy.deepcopy(self)
        cloned_adapter.quantized = self.quantized
        cloned_adapter.device = self.device
        cloned_adapter.model.load_state_dict(self.model.state_dict())
        cloned_adapter.model.to(self.device)
        return cloned_adapter

    def eval(self):
        self.model.eval()

    def _evaluate(self,
                 dataset: Dataset,
                 criterion: nn.modules.loss._WeightedLoss,
                 batch_size: int = 32,
                 max_batches: int = None,
                 top: Union[List[int], int] = [1]
                 ):
        original_train_mode = self.model.training
        top = top if isinstance(top, list) else [top]
        try:
            self.model.eval()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            running_loss = 0
            total_samples = 0
            running_correct = 0
            with torch.inference_mode():
                for batch_index, (x, y) in enumerate(loader):
                    x.to(self.device)
                    y_hat = self.model(x)
                    loss = criterion(y_hat, y)
                    running_loss += loss.item() * y.size(0)

                    for t in top:
                        _, predicted = y_hat.topk(top, dim=1)
                    correct = predicted.eq(y.view(-1, 1).expand_as(predicted))
                    running_correct += correct.sum().item()
                    total_samples += y.size(0)

                    if batch_index % 100 == 0:
                        print(f"batch {batch_index} acc: {100 * correct / y.size(0):.12f}%, loss: {loss / y.size(0):.12f}")
                    if max_batches is not None and batch_index >= max_batches:
                        break

            acc = 100 * running_correct / total_samples
            average_loss = running_loss / total_samples
            return acc, average_loss
        except Exception as e:
            raise e
        finally:
            self.model.train(mode=original_train_mode)

    def get_bias(self, layer: Union[nn.Module, str]):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer
        if self.quantized:
            return layer.bias()
        else:
            return layer.bias
        
    def get_weights(self, layer: Union[nn.Module, str]):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer
        if self.quantized:
            return layer.weight().detach()
        else:
            return layer.weight.detach()

    def get_train_weights(self, layer: Union[nn.Module, str]):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer
        if self.quantized:
            return layer.weight().detach().int_repr()
        else:
            print(self.quantized)
            print(self.model.__class__.__name__)
            return layer.weight.detach()        

    def set_weights_bias(self, layer: Union[nn.Module, str], weights: torch.Tensor, biases: torch.Tensor):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer

        if self.quantized:
            layer.set_weight_bias(weights, biases)
        else:
            layer.weight = weights 
            layer.bias = biases

    def inject_weights(self, weights_vector: List[torch.Tensor], all_injection_plans: List[List[FilterSelector]]):
        original_train_mode = self.model.training
        model = self.clone()
        try:
            model.eval()
            with torch.inference_mode():
                offset = 0
                for weights, injection_plans in zip(weights_vector, all_injection_plans):
                    for plan in injection_plans:
                        bias = self.get_bias(plan.layer_name)
                        fp32_weights = self.get_weights(plan.layer_name)

                        for out_selector in plan.out:
                            initial_output_tensor = fp32_weights[*out_selector]
                            size = None
                            if isinstance(out_selector[0], slice) and out_selector[0].start is None and out_selector[0].stop is None:
                                for filter_i, filter_tensor in enumerate(initial_output_tensor):
                                    if isinstance(out_selector[1], slice) and out_selector[1].start is None and out_selector[1].stop is None:
                                        for channel_tensor_i, channel_tensor in enumerate(filter_tensor):
                                            w = fp32_weights[filter_i, channel_tensor_i, *out_selector[2:]]
                                            size = reduce(operator.mul, w.shape)
                                            fp32_weights[filter_i, channel_tensor_i, *out_selector[2:]] = dequantize_per_tensor(weights[offset:offset+size], w.q_scale(), w.q_zero_point())
                                    else:
                                        w = fp32_weights[filter_i, out_selector[1], *out_selector[2:]]
                                        size = reduce(operator.mul, w.shape)
                                        fp32_weights[filter_i, out_selector[1], *out_selector[2:]] = dequantize_per_tensor(weights[offset:offset+size], w.q_scale(), w.q_zero_point())
                            else:
                                w = initial_output_tensor
                                size = reduce(operator.mul, w.shape)
                                fp32_weights[*out_selector] = dequantize_per_tensor(weights[offset:offset+size], w.q_scale(), w.q_zero_point())
                            offset += size
                        self.set_weights_bias(getattr(model, plan.layer_name), fp32_weights, bias)
                return model
        finally:
            model.train(mode=original_train_mode)
