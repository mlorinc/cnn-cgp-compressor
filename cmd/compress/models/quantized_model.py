import copy
import torch
from models.base_model import BaseModel
from abc import ABC, abstractmethod
from typing import Self, Optional
from tqdm import tqdm

class QuantizedBaseModel(BaseModel, ABC):
    """
    Abstract base class for quantized models.

    Attributes:
        prepared (bool): Flag indicating whether the model is prepared for quantization.
    """    
    def __init__(self, model_path: str = None):
        """
        Initialize the QuantizedBaseModel.

        Args:
            model_path (str, optional): Path to the model.
        """        
        super().__init__(model_path)
        self.prepared = False

    def clone(self) -> Self:
        """
        Clone the model.

        Returns:
            QuantizedBaseModel: Cloned model.
        """        
        clone = self._create_self()
        if self.prepared:
            clone._prepare()
            clone._convert()
        clone.load_state(self.get_state())
        return clone

    @abstractmethod
    def _prepare(self):
        """
        Abstract method to prepare the model for quantization.
        """        
        self.prepared = True

    @abstractmethod
    def _convert(self):
        """
        Abstract method to convert the prepared model to quantized form.
        """        
        self.prepared = True
    
    @abstractmethod
    def quantize(self, new_path: str = None, inline=True) -> Self:
        """
        Abstract method to quantize the model.

        Args:
            new_path (str, optional): Path to save the quantized model.
            inline (bool, optional): Whether to save the quantized model inline or not.

        Returns:
            QuantizedBaseModel: Quantized model.
        """        
        raise NotImplementedError()

    def load_unquantized(self, model_path: Optional[str] = None):
        """
        Load the unquantized model.

        Args:
            model_path (str, optional): Path to the unquantized model.

        Returns:
            QuantizedBaseModel: Unquantized model.
        """        
        return super().load(model_path)

    def load(self, model_path: Optional[str] = None):
        """
        Load the model.

        Args:
            model_path (str, optional): Path to the model.

        Returns:
            QuantizedBaseModel: Loaded model.
        """        
        self.eval()
        self._prepare()
        self._convert()
        super().load(model_path)
        return self

    def ptq_quantization(self) -> Self:
        """
        Perform post-training quantization on the model using PyTorch Quantization.

        Returns:
            QuantizedBaseModel: Quantized model.
        """        
        reference_model = copy.deepcopy(self)
        self.eval()
        self._prepare()

        # Calibrate
        with torch.inference_mode():
            _, val_loader = self.get_split_train_validation_loaders()
            with tqdm(val_loader, unit="Batch", total=len(val_loader), leave=True) as loader:
                for x, _ in loader:
                    self(x)
    
        self._convert()

        # 1 byte instead of 4 bytes for FP32
        assert self.conv1.weight().element_size() == 1
        assert self.conv2.weight().element_size() == 1
        assert reference_model.conv1.weight.element_size() == 4

    def qat_quantization(self):
        """
        Perform quantization aware training (QAT) on the model.

        Returns:
            QuantizedBaseModel: Quantized model.
        """        
        reference_model = copy.deepcopy(self)

        self.eval()
        self._prepare()
        self.fit()
        self.eval()
        self._convert()

        # 1 byte instead of 4 bytes for FP32
        assert self.conv1.weight().element_size() == 1
        assert self.conv2.weight().element_size() == 1
        assert reference_model.conv1.weight.element_size() == 4        