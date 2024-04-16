import torch
import torch.nn as nn
import operator
import copy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import List, Union, Self, Iterable, Optional
from functools import reduce
from models.base_model import BaseModel
from models.quantization import dequantize_per_tensor
from models.selector import FilterSelector
from tqdm import tqdm

class ModelAdapter(ABC):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @abstractmethod
    def load(self, path: str = None, inline: Optional[bool] = True) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_criterion(self, **kwargs):
        raise NotImplementedError()

    def get_layer(self, layer: str) -> nn.Module:
        return getattr(self.model, layer)

    def clone(self):
        cloned_adapter = copy.deepcopy(self)
        cloned_adapter.device = self.device

        if isinstance(cloned_adapter.model, BaseModel) and isinstance(self.model, BaseModel):
            cloned_adapter.model.load_state(self.model.get_state())
        else:
            cloned_adapter.model.load_state(self.model.state_dict())

        cloned_adapter.model.to(self.device)
        return cloned_adapter

    def eval(self):
        self.model.eval()

    def evaluate(self,
                 batch_size: int = 32,
                 max_batches: int = None,
                 top: Union[List[int], int] = 1,
                 include_loss: bool =True,
                 show_top_k: int = 2
                 ):
        top = set([1] + top) if isinstance(top, Iterable) else set([1, top])
        original_train_mode = self.model.training
        dataset = self.get_test_data()
        criterion = self.get_criterion()
        try:
            self.model.eval()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            running_loss = 0
            total_samples = 0
            running_topk_correct = dict([(k, 0) for k in top])
            with torch.inference_mode():
                with tqdm(enumerate(loader), unit="batch", total=len(loader), leave=True) as pbar:
                    for batch_index, (x, y) in pbar:
                        y_hat = self.model(x)

                        if include_loss:
                            loss = criterion(y_hat, y)
                            running_loss += loss.item() * y.size(0)
                        
                        for k in top:
                            _, predicted = y_hat.topk(k, dim=1)
                            correct = predicted.eq(y.view(-1, 1).expand_as(predicted))
                            running_topk_correct[k] += correct[:, :k].sum().item()
                        total_samples += y.size(0)

                        top_k = {k: 100 * v / total_samples for k, v in running_topk_correct.items()}
                        top_k_strings = [f"Top-{k}: {v:.6f}%" for k, v in running_topk_correct.items()[1:show_top_k]]
                        top_k_strings = (", " + ", ".join(top_k_strings) if top_k_strings else "")
                        if include_loss:
                            pbar.set_description(f"Loss: {running_loss / total_samples:.4f}, Acc: {top_k[1]:.6f}" + top_k_strings)
                        else:
                            pbar.set_description(f"Acc: {top_k[1]:.6f}" + top_k_strings)

                        if max_batches is not None and batch_index >= max_batches:
                            break
                                

            top_k = {k: 100 * v / total_samples for k, v in running_topk_correct.items()}      
            average_loss = running_loss / total_samples

            if len(top_k) == 1:
                return top_k[1], average_loss
            else:
                return top_k, average_loss
        finally:
            self.model.train(mode=original_train_mode)
        
    def get_bias(self, layer: Union[nn.Module, str]):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer
        try:
            return layer.bias()
        except:
            return layer.bias
        
    def get_weights(self, layer: Union[nn.Module, str]):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer
        try:
            return layer.weight().detach()
        except:
            return layer.weight.detach()

    def get_train_weights(self, layer: Union[nn.Module, str]):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer
        try:
            return layer.weight().detach().int_repr()
        except:
            return layer.weight.detach()        

    def set_weights_bias(self, layer: Union[nn.Module, str], weights: torch.Tensor, biases: torch.Tensor):
        layer = getattr(self.model, layer) if isinstance(layer, str) else layer

        try:
            layer.set_weight_bias(weights, biases)
        except:
            layer.weight = weights 
            layer.bias = biases

    def inject_weights(self, weights_vector: List[torch.Tensor], all_injection_plans: List[List[FilterSelector]], inline=False):
        original_train_mode = self.model.training
        model = self.clone() if not inline else self
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
                        self.set_weights_bias(plan.layer_name, fp32_weights, bias)
                return model
        finally:
            model.train(mode=original_train_mode)