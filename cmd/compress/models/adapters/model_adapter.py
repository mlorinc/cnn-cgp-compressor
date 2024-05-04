import torch
import torch.nn as nn
import operator
import copy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import List, Union, Self, Iterable, Optional, Callable
from functools import reduce
from models.adapters.model_adapter_interface import ModelAdapterInterface
from models.base_model import BaseModel
from models.quantization import dequantize_per_tensor, tensor_iterator
from models.selector import FilterSelectorCombinations
from tqdm import tqdm

class ModelAdapter(ModelAdapterInterface, ABC):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    @abstractmethod
    def load(self, path: str = None, inline: Optional[bool] = True) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_criterion(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_convolution_layers(self) -> Iterable[nn.Conv2d]:
        raise NotImplementedError()

    def get_layer(self, selector: Union[str, Callable[[Self], nn.Conv2d]]) -> nn.Module:
        if isinstance(selector, str):
            return getattr(self.model, selector)
        elif isinstance(selector, nn.Module):
            return selector
        else:
            return selector(self)

    @abstractmethod
    def clone(self):
        cloned_adapter = copy.deepcopy(self)
        cloned_adapter.device = self.device
        cloned_adapter.model = copy.deepcopy(self.model)
        if isinstance(cloned_adapter.model, BaseModel) and isinstance(self.model, BaseModel):
            cloned_adapter.model.load_state(self.model.get_state())
        else:
            cloned_adapter.model.load_state_dict(self.model.state_dict())

        if hasattr(cloned_adapter.model, "backend"):
            cloned_adapter.model.backend = self.model.backend
        cloned_adapter.model.to(self.device)
        return cloned_adapter

    def eval(self):
        self.model.eval()

    def train(self, mode=True):
        self.model.train(mode=True)

    def evaluate(self,
                 batch_size: int = None,
                 max_batches: int = None,
                 top: Union[List[int], int] = 1,
                 include_loss: bool =True,
                 show_top_k: int = 2,
                 num_workers: int = 1,
                 **kwargs
                 ):
        top = set([1] + top) if isinstance(top, Iterable) else set([1, top])
        original_train_mode = self.model.training
        dataset = self.get_test_data(**kwargs)
        criterion = self.get_criterion(**kwargs)
        try:
            self.model.eval()
            loader = DataLoader(dataset, batch_size=batch_size or len(dataset), shuffle=False, num_workers=num_workers)
            running_loss = 0
            total_samples = 0
            running_topk_correct = dict([(k, 0) for k in top])
            with torch.inference_mode():
                with tqdm(enumerate(loader), unit="batch", total=len(loader), leave=True) as pbar:
                    for batch_index, (x, y) in pbar:
                        y_hat = self.model(x)

                        if include_loss and criterion is not None:
                            loss = criterion(y_hat, y)
                            running_loss += loss.item() * y.size(0)
                        
                        for k in top:
                            _, predicted = y_hat.topk(k, dim=1)
                            correct = predicted.eq(y.view(-1, 1).expand_as(predicted))
                            running_topk_correct[k] += correct[:, :k].sum().item()
                        total_samples += y.size(0)

                        top_k = {k: v / total_samples for k, v in running_topk_correct.items()}
                        top_k_strings = [f"Top-{k}: {v:.6f}" for k, v in list(top_k.items())[1:show_top_k]]
                        top_k_strings = (", " + ", ".join(top_k_strings) if top_k_strings else "")
                        if include_loss and criterion is not None:
                            pbar.set_description(f"Loss: {running_loss / total_samples:.4f}, Acc: {top_k[1]:.6f}" + top_k_strings)
                        else:
                            pbar.set_description(f"Acc: {top_k[1]:.6f}" + top_k_strings)

                        if max_batches is not None and batch_index >= max_batches:
                            break
                                

            top_k = {k: v / total_samples for k, v in running_topk_correct.items()}      
            average_loss = running_loss / total_samples

            if len(top_k) == 1:
                return top_k[1], average_loss
            else:
                return top_k, average_loss
        finally:
            self.model.train(mode=original_train_mode)
        
    def get_bias(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        layer = self.get_layer(layer)
        try:
            return layer.bias()
        except:
            return layer.bias
        
    def get_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        layer = self.get_layer(layer)
        try:
            return layer.weight().detach()
        except:
            return layer.weight.detach()

    def get_train_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        layer = self.get_layer(layer)
        try:
            return layer.weight().detach().int_repr()
        except:
            return layer.weight.detach()        

    def set_weights_bias(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]], weights: torch.Tensor, biases: torch.Tensor):
        layer = self.get_layer(layer)
        try:
            layer.set_weight_bias(weights, biases)
        except Exception as e:
            layer.weight = weights 
            layer.bias = biases

    def inject_weights(self, weights_vector: List[torch.Tensor], injection_combinations: FilterSelectorCombinations, inline=False):
        original_train_mode = self.model.training
        model = self.clone() if not inline else self
        try:
            model.eval()
            with torch.inference_mode():
                for weights, injection_plans in zip(weights_vector, injection_combinations.get_combinations()):
                    offset = 0
                    for sel in injection_plans.get_selectors():
                        bias = model.get_bias(sel.selector)
                        fp32_weights = model.get_weights(sel.selector)
                        for w, size, out_selector in tensor_iterator(fp32_weights, sel.out):
                            # print("Layer:", plan.layer_name, "Sel:", out_selector, "Size:", size)
                            fp32_weights[*out_selector] = dequantize_per_tensor(weights[offset:offset+size], w.q_scale(), w.q_zero_point())                                      
                            offset += size
                        model.set_weights_bias(sel.selector, fp32_weights, bias)
                    print("offset:", offset, "size:", reduce(operator.mul, weights.shape))
                    # assert offset == reduce(operator.mul, weights.shape)
                return model
        finally:
            model.train(mode=original_train_mode)
