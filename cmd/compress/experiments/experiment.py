import torch
import operator
import glob
from functools import reduce
from abc import abstractmethod
from cgp.cgp_adapter import CGP
from typing import List, Tuple, Union, Iterable
from models.base import BaseModel
from models.lenet import LeNet5
from pathlib import Path

ExperimentRange = Union[Tuple[int, int], int]
InputRange = Union[ExperimentRange, Iterable[Tuple[int, int]]]
OutputRange = Union[InputRange, None]

class CGPPinPlanner(object):
    def __init__(self) -> None:
        self._plan = []
        self._preliminary_plan = []
    def add_mapping(self, layer_name: str, input_selectors, output_selectors):
        self._preliminary_plan.append((layer_name, input_selectors, output_selectors))
    def next_mapping(self):
        self._plan.append(self._preliminary_plan[:])
        self._preliminary_plan.clear()
    def finish_mapping(self):
        if self._preliminary_plan:
            self._plan.append(self._preliminary_plan[:])
            self._preliminary_plan.clear()
    def get_plan(self, i: int = None):
        if i is None:
            return iter(self._plan)
        return iter(self._plan[i])

class Experiment(object):
    def __init__(self, experiment_name: str, model: BaseModel, cgp: CGP, dtype=torch.int8) -> None:
        self.experiment_name = experiment_name
        self._model = model
        self._cgp = cgp
        self._planner = CGPPinPlanner() 
        self.dtype = dtype
        self._cgp_prepared = False
        self._to_number = int if self.dtype == torch.int8 else float

    def next_input_combination(self):
        self._planner.next_mapping()

    def add_filters(
            self,
            layer_name: str,
            input_selectors,
            output_selectors
        ):
        self._planner.add_mapping(layer_name, input_selectors, output_selectors)

    def _prepare_cgp(self):
        if self._cgp_prepared: 
            return
        try:
            self._model.eval()
            self._planner.finish_mapping()
            with torch.inference_mode():
                for plan in self._planner.get_plan():
                    for layer_name, input_selectors, output_selectors in plan:
                        weights = self._get_train_weights(layer_name)
                        for in_selector in input_selectors:
                            self._cgp.add_inputs(weights[*in_selector])
                        for out_selector in output_selectors:
                            self._cgp.add_outputs(weights[*out_selector])
                    self._cgp.next_train_item()
            self._cgp_prepared = True
        finally:
            self._model.train(True)

    @abstractmethod
    def _get_weight_output_file(self) -> str:
        pass

    @abstractmethod
    def _get_cgp_output_file(self) -> str:
        pass

    @abstractmethod
    def _get_cgp_output_file(self) -> str:
        pass

    @abstractmethod
    def _get_train_file(self) -> str:
        pass

    @abstractmethod
    def _get_statistics_file(self) -> str:
        pass

    @abstractmethod
    def get_number_of_experiment_results(self) -> int:
        pass

    def execute(self):
        self._prepare_cgp()
        config = self._cgp.config.clone()
        config.set_output_file(self._get_cgp_output_file())
        config.set_input_file(self._get_train_file())
        config.set_cgp_statistics_file(self._get_statistics_file())
        self._cgp.train(config)

    def evaluate(self, run: int = 0):
        self._prepare_cgp()
        with torch.inference_mode():
            config = self._cgp.config.clone()
            config.set_output_file(f"{self._get_weight_output_file()}.{run}")
            self._cgp.evaluate(new_configration=config, config_file=f"{self._get_cgp_output_file()}.{run}")     
            new_model = self._inject_weights_from_file(run=run)
            after_acc, after_loss = new_model.evaluate()
            return after_acc, after_loss

    def evaluate_from_file(self, file: str = None, run: int = 0):
        self._planner.finish_mapping()
        with torch.inference_mode():
            new_model = self._inject_weights_from_file(file=file, run=run)
            after_acc, after_loss = new_model.evaluate()
            return after_acc, after_loss

    def evaluate_from_solution(self, solution: str, file: str):
        self._prepare_cgp()
        with torch.inference_mode():
            config = self._cgp.config.clone()
            config.set_output_file(file)
            self._cgp.evaluate(new_configration=config, solution=solution)     
            new_model = self._inject_weights_from_file(file=file)
            after_acc, after_loss = new_model.evaluate()
            return after_acc, after_loss

    def _get_bias(self, layer_name: str):
        if self.dtype == torch.int8:
            return getattr(self._model, layer_name).bias()
        else:
            return getattr(self._model, layer_name).bias
        
    def _get_weights(self, layer_name: str):
        if self.dtype == torch.int8:
            return getattr(self._model, layer_name).weight().detach()
        else:
            return getattr(self._model, layer_name).weight.detach()

    def _get_train_weights(self, layer_name: str):
        if self.dtype == torch.int8:
            return getattr(self._model, layer_name).weight().detach().int_repr()
        else:
            return getattr(self._model, layer_name).weight.detach()        

    def _get_reconstruction_weights(self, layer_name: str):
        if self.dtype == torch.int8:
            return getattr(self._model, layer_name).weight().detach()
        else:
            return getattr(self._model, layer_name).weight.detach()    

    def _inject_weights(self, weights_vector: List[torch.Tensor]):
        model = self._model.clone()
        model.eval()
        offset = 0
        for weights, plan in zip(weights_vector, self._planner.get_plan()):
            for layer_name, _, output_selectors in plan:
                bias = self._get_bias(layer_name)
                fp32_weights = self._get_reconstruction_weights(layer_name)

                for out_selector in output_selectors:
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
                self._set_weights_bias(getattr(model, layer_name), fp32_weights, bias)
        return model

    def _inject_weights_from_file(self, file: str = None, run: int = 0):
        file = file or f"{self._get_weight_output_file()}.{run}"
        weights_vector = []
        with open(file) as f:
            for line, _ in zip(f.readlines(), range(self._cgp.config.get_dataset_size())):
                segments = line.split(" ")

                if "nan" in segments:
                    raise ValueError(f"CGP training failed for {file}; the file contains invalid weight")

                weights = torch.Tensor([self._to_number(segment) for segment in segments if segment.strip() != ""])
                weights_vector.append(weights)
        return self._inject_weights(weights_vector)

    def _set_weights_bias(self, layer, weights, biases):
        if self.dtype == torch.int8:
            layer.set_weight_bias(weights, biases)
        else:
            layer.weight = weights 
            layer.bias = biases

class BaseExperiment(Experiment):
    def __init__(self, experiment_folder: str, experiment_name: str, model: BaseModel, cgp: CGP, dtype=torch.int8) -> None:
        super().__init__(experiment_name, model, cgp, dtype)
        self.experiment_folder_path = Path(experiment_folder)
        self.experiment_folder_path.mkdir(exist_ok=True, parents=True)
        self.experiment_root_path = self.experiment_folder_path / experiment_name
        self.configs = self.experiment_root_path / "cgp_configs"
        self.weights = self.experiment_root_path / "weights"
    
    def _get_cgp_output_file(self) -> str:
        return self.configs / "data.cgp"
    
    def _get_weight_output_file(self) -> str:
        return self.weights / "inferred_weights"
    
    def _get_train_file(self) -> str:
        return self.experiment_root_path / "train.data"

    def _get_statistics_file(self) -> str:
        return self.experiment_root_path / "statistics.csv"

    def get_number_of_experiment_results(self) -> int:
        return len(glob.glob(f"{str(self._get_cgp_output_file())}.*"))

    def execute(self):
        self.experiment_root_path.mkdir(exist_ok=False)
        self.configs.mkdir(exist_ok=False, parents=True)
        self.weights.mkdir(exist_ok=False, parents=True)
        return super().execute()


def conv2d_core_slices(kernel_size, core_size):
    # Ensure the core size is valid
    if core_size % 2 == 0 and kernel_size == core_size:
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")

    skip = (kernel_size - core_size) // 2
    c = slice(skip, skip + core_size)
    # Extract the core
    return [c, c]

def conv2d_outter_slices(kernel_size, core_size):
    # Ensure the core size is valid
    if core_size % 2 == 0 and kernel_size == core_size:
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")
    skip = (kernel_size - core_size) // 2

    output_indices = []
    row = 0
    for _ in range(skip):
        output_indices.append((row, slice(None)))
        row += 1

    for _ in range(core_size):
        output_indices.append((row, slice(0, skip)))
        output_indices.append((row, slice(skip+core_size, None)))
        row += 1
    
    for _ in range(skip):
        output_indices.append((row, slice(None)))
        row += 1
    return output_indices

def conv2d_outter(selectors, kernel_size, core_size):
    outter = conv2d_outter_slices(kernel_size, core_size)
    slices = []
    for out in outter:
        slices.append((*selectors, *out))
    return slices

def conv2d_core(selectors, kernel_size, core_size):
    core = conv2d_core_slices(kernel_size, core_size)
    return (*selectors, *core)

def dequantize_per_channel(x: torch.Tensor, conv_layer: torch.Tensor):
    zero_point = conv_layer.q_per_channel_zero_points()
    scale = conv_layer.q_per_channel_scales()

    dequantized = ((x - zero_point.view(-1, 1, 1)) * scale.view(-1, 1, 1)).float()
    return torch.quantize_per_channel(
        dequantized,
        scale,
        zero_point,
        axis=0,
        dtype=torch.qint8
    )

def dequantize_per_tensor(x: torch.Tensor, scale: torch.float32, zero_point: torch.float32):
    dequantized = ((x - zero_point) * scale).float()
    return torch.quantize_per_tensor(
        dequantized,
        scale,
        zero_point,
        dtype=torch.qint8
    )
