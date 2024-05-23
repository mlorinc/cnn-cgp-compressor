import torch
import torch.nn as nn
import operator
import copy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import random
from typing import List, Union, Self, Iterable, Optional, Callable
from functools import reduce
from models.adapters.model_adapter_interface import ModelAdapterInterface
from models.base_model import BaseModel
from models.quantization import quantize_per_tensor, tensor_iterator
from models.selector import FilterSelectorCombinations
from tqdm import tqdm

class ModelAdapter(ModelAdapterInterface, ABC):
    """
    Abstract base class for model adapters, providing a framework for model evaluation,
    loading, saving, and weight injection.

    Attributes:
        model (nn.Module): The neural network model to be adapted.
        device (str): The device on which the model is loaded (either 'cuda' or 'cpu').
    """    
    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the ModelAdapter with the provided model and set the device to GPU if available,
        otherwise to CPU.

        Args:
            model (nn.Module): The neural network model to be adapted.
        """        
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def save(self, path: str):
        """
        Save the model's state dictionary to the specified file path.

        Args:
            path (str): The file path where the model state dictionary will be saved.
        """        
        torch.save(self.model.state_dict(), path)

    @abstractmethod
    def load(self, path: str = None, inline: Optional[bool] = True) -> Self:
        """
        Load the model's state dictionary from the specified file path.

        Args:
            path (str, optional): The file path from which to load the model state dictionary.
            inline (bool, optional): Whether to load the state dictionary inline or not.

        Returns:
            Self: The loaded model adapter instance.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        """
        Get the test dataset for the model.

        Returns:
            Dataset: The test dataset.
        """        
        raise NotImplementedError()

    def get_custom_dataset(self, **kwargs):
        """
        Get a custom dataset.

        Returns:
            Dataset: The custom dataset.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_criterion(self, **kwargs):
        """
        Get the loss criterion for the model.

        Returns:
            Criterion: The loss criterion.
        """        
        raise NotImplementedError()

    @abstractmethod
    def get_convolution_layers(self) -> Iterable[nn.Conv2d]:
        """
        Get all the convolutional layers of the model.

        Returns:
            Iterable[nn.Conv2d]: An iterable of convolutional layers.
        """        
        raise NotImplementedError()

    @abstractmethod
    def clone(self):
        """
        Create a deep copy of the model adapter instance.

        Returns:
            ModelAdapter: A deep copy of the model adapter instance.
        """        
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
        """
        Set the model to evaluation mode.
        """        
        self.model.eval()

    def train(self, mode=True):
        """
        Set the model to training mode.

        Args:
            mode (bool, optional): Whether to set the model to training mode. Defaults to True.
        """        
        self.model.train(mode=True)

    def evaluate(self,
                 batch_size: int = None,
                 max_batches: int = None,
                 top: Union[List[int], int] = 1,
                 include_loss: bool =True,
                 show_top_k: int = 2,
                 num_workers: int = 1,
                 custom_dataset=False,
                 **kwargs
                 ):
        """
        Evaluate the model on the test dataset.

        Args:
            batch_size (int, optional): The batch size for evaluation. Defaults to None.
            max_batches (int, optional): The maximum number of batches to evaluate. Defaults to None.
            top (Union[List[int], int], optional): The top-k accuracy to compute. Defaults to 1.
            include_loss (bool, optional): Whether to include loss in the evaluation. Defaults to True.
            show_top_k (int, optional): The number of top-k accuracies to display. Defaults to 2.
            num_workers (int, optional): The number of workers for data loading. Defaults to 1.
            custom_dataset (bool, optional): Whether to use a custom dataset. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[float, Tuple[Dict[int, float], float]]: The top-1 accuracy and loss if `top` is an int,
                otherwise a dictionary of top-k accuracies and loss.
        """        
        top = set([1] + top) if isinstance(top, Iterable) else set([1, top])
        original_train_mode = self.model.training
        dataset = self.get_test_data(**kwargs) if not custom_dataset else self.get_custom_dataset(**kwargs)
        criterion = self.get_criterion(**kwargs)
        print(f"dataset has {len(dataset)} samples")
        try:
            self.model.eval()
            loader = DataLoader(dataset, batch_size=batch_size or len(dataset), shuffle=False, num_workers=num_workers or 0)
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
            top_k_strings = [f"Top-{k}: {v:.6f}" for k, v in list(top_k.items())[1:show_top_k]]
            top_k_strings = (", " + ", ".join(top_k_strings) if top_k_strings else "")
            average_loss = running_loss / total_samples
            if include_loss and criterion is not None:
                print(f"Loss: {average_loss:.4f}, Acc: {top_k[1]:.6f}" + top_k_strings)
            else:
                print(f"Acc: {top_k[1]:.6f}" + top_k_strings)
            if len(top_k) == 1:
                return top_k[1], average_loss
            else:
                return top_k, average_loss
        finally:
            self.model.train(mode=original_train_mode)
        
    def get_bias(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the bias of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The bias tensor.
        """        
        layer = self.get_layer(layer)
        try:
            return layer.bias()
        except:
            return layer.bias
        
    def get_layer(self, selector: Union[str, Callable[[Self], nn.Conv2d]]) -> nn.Module:
        """
        Get the specified layer.

        Args:
            selector (Union[str, Callable[[Self], nn.Conv2d]]): The layer name or a function to get the layer.

        Returns:
            nn.Module: The specified layer.
        """        
        if isinstance(selector, str):
            return getattr(self, selector)
        elif isinstance(selector, nn.Module):
            return selector
        else:
            return selector(self)

    def _get_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the weights of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The weights tensor.
        """        
        layer = self.get_layer(layer)
        try:
            return layer.weight().detach()
        except:
            return layer.weight.detach() 

    def get_weights(self, selector: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the weights of the specified layer.

        Args:
            selector (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The weights tensor.
        """        
        if not isinstance(selector, str):
            return self._get_weights(selector)
        return self.model.state_dict()[selector + ".weight"]

    def get_train_weights(self, selector: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the train weights of the specified layer.

        Args:
            selector (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The train weights tensor.
        """        
        return self.get_weights(selector).int_repr()      

    def _set_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]], weights: torch.Tensor):
        """
        Set the weights of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.
            weights (torch.Tensor): The weights tensor to be set.
        """        
        layer = self.get_layer(layer)
        try:
            layer.set_weight(weights)
        except Exception as e:
            layer.weight = weights 
            
    def set_weights(self, layer: str, weights: torch.Tensor):
        """
        Set the weights of the specified layer.

        Args:
            layer (str): The name of the layer.
            weights (torch.Tensor): The weights tensor to be set.
        """        
        if not isinstance(layer, str):
            self._set_weights(layer, weights)
        else:
            state_dict = self.model.state_dict()
            state_dict[layer + ".weight"] = weights
            self.model.load_state_dict(state_dict)            
            

    def inject_weights(self, weights_vector: List[torch.Tensor], injection_combinations: FilterSelectorCombinations, inline=False, debug=False):
        """
        Inject the specified weights into the model according to created plan.

        Args:
            weights_vector (List[torch.Tensor]): A list of weight tensors to be injected.
            injection_combinations (FilterSelectorCombinations): The filter selector combinations for weight injection forming injection plan.
            inline (bool, optional): Whether to inject weights inline. Defaults to False.
            debug (bool, optional): Whether to inject random weights for debugging. Defaults to False.

        Returns:
            ModelAdapter: The model adapter with injected weights.
        """        
        original_train_mode = self.model.training
        model = self.clone() if not inline else self
        try:
            model.eval()
            with torch.inference_mode():
                for weights, injection_plans in zip(weights_vector, injection_combinations.get_combinations()):
                    offset = 0
                    for sel in injection_plans.get_selectors():
                        fp32_weights = model.get_weights(sel.selector)
                        for w, size, out_selector in tensor_iterator(fp32_weights, sel.out):
                            # print("Layer:", plan.layer_name, "Sel:", out_selector, "Size:", size)
                            if not debug:
                                fp32_weights[*out_selector] = quantize_per_tensor(weights[offset:offset+size], w.q_scale(), w.q_zero_point())
                            else:
                                fp32_weights[*out_selector] = quantize_per_tensor(torch.tensor([random.randint(-128, 127) for _ in range(size)], dtype=torch.int8), w.q_scale(), w.q_zero_point())                                      
                            offset += size
                        model.set_weights(sel.selector, fp32_weights)
                    print("offset:", offset, "size:", reduce(operator.mul, weights.shape))
                    # assert offset == reduce(operator.mul, weights.shape)
                return model
        finally:
            model.train(mode=original_train_mode)
