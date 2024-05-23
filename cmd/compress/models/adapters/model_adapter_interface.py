from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Union, Self, Optional, Callable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.selector import FilterSelectorCombinations

class ModelAdapterInterface(ABC):
    """
    Abstract base class for model adapter interfaces, defining the necessary methods for model interaction.

    This interface includes methods for loading, evaluating, and manipulating model weights, as well as getting data and criterion.
    """

    @abstractmethod
    def load(self, path: str = None, inline: Optional[bool] = True):
        """
        Load the model's state dictionary from the specified file path.

        Args:
            path (str, optional): The file path from which to load the model state dictionary.
            inline (bool, optional): Whether to load the state dictionary inline or not.
        """
        pass

    @abstractmethod
    def get_test_data(self, **kwargs):
        """
        Get the test dataset for the model.

        Returns:
            Dataset: The test dataset.
        """
        pass

    @abstractmethod
    def get_criterion(self, **kwargs):
        """
        Get the loss criterion for the model.

        Returns:
            Criterion: The loss criterion.
        """
        pass

    @abstractmethod
    def clone(self):
        """
        Create a deep copy of the model adapter instance.

        Returns:
            ModelAdapterInterface: A deep copy of the model adapter instance.
        """
        pass

    @abstractmethod
    def eval(self):
        """
        Set the model to evaluation mode.
        """
        pass

    @abstractmethod
    def train(self, mode=True):
        """
        Set the model to training mode.

        Args:
            mode (bool, optional): Whether to set the model to training mode. Defaults to True.
        """
        pass

    @abstractmethod
    def evaluate(self,
                 batch_size: int = None,
                 max_batches: int = None,
                 top: Union[List[int], int] = 1,
                 include_loss: bool = True,
                 show_top_k: int = 2):
        """
        Evaluate the model on the test dataset.

        Args:
            batch_size (int, optional): The batch size for evaluation. Defaults to None.
            max_batches (int, optional): The maximum number of batches to evaluate. Defaults to None.
            top (Union[List[int], int], optional): The top-k accuracy to compute. Defaults to 1.
            include_loss (bool, optional): Whether to include loss in the evaluation. Defaults to True.
            show_top_k (int, optional): The number of top-k accuracies to display. Defaults to 2.

        Returns:
            Union[float, Tuple[Dict[int, float], float]]: The top-1 accuracy and loss if `top` is an int,
                otherwise a dictionary of top-k accuracies and loss.
        """
        pass

    @abstractmethod
    def get_bias(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the bias of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The bias tensor.
        """
        pass

    @abstractmethod
    def get_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the weights of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The weights tensor.
        """
        pass

    @abstractmethod
    def get_train_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]]):
        """
        Get the train weights of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.

        Returns:
            Tensor: The train weights tensor.
        """
        pass

    @abstractmethod
    def set_weights(self, layer: Union[nn.Module, str, Callable[[Self], nn.Conv2d]], weights: torch.Tensor):
        """
        Set the weights of the specified layer.

        Args:
            layer (Union[nn.Module, str, Callable[[Self], nn.Conv2d]]): The layer or its name or a function to get the layer.
            weights (torch.Tensor): The weights tensor to be set.
        """
        pass

    @abstractmethod
    def inject_weights(self, weights_vector: List[torch.Tensor], injection_combinations: FilterSelectorCombinations, inline=False):
        """
        Inject the specified weights into the model according to created plan.

        Args:
            weights_vector (List[torch.Tensor]): A list of weight tensors to be injected.
            injection_combinations (FilterSelectorCombinations): The filter selector combinations for weight injection forming injection plan.
            inline (bool, optional): Whether to inject weights inline. Defaults to False.
            debug (bool, optional): Whether to inject random weights for debugging. Defaults to False.

        Returns:
            Self: The model adapter with injected weights.
        """   
        pass
