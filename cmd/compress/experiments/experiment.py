import os
import operator
import torch
import pandas as pd
import shutil
from functools import reduce
from pathlib import Path
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from typing import List, Tuple, Union, Iterable, Self, Dict, Optional
from models.base import BaseModel


class FilterSelector(object):
    def __init__(self, layer_name: str, inp: List, out: List) -> None:
        self.layer_name = layer_name
        self.inp = inp
        self.out = out

class CGPPinPlanner(object):
    def __init__(self) -> None:
        self._plan: List[List[FilterSelector]] = []
        self._preliminary_plan: List[FilterSelector] = []

    def clone(self):
        planner = CGPPinPlanner()
        planner._plan = self._plan[:]
        planner._preliminary_plan = self._preliminary_plan[:]
        
    def add_mapping(self, sel: FilterSelector):
        self._preliminary_plan.append(sel)
    def next_mapping(self):
        self._plan.append(self._preliminary_plan[:])
        self._preliminary_plan.clear()
    def finish_mapping(self):
        if self._preliminary_plan:
            self._plan.append(self._preliminary_plan[:])
            self._preliminary_plan.clear()
    def get_plan(self):
        return iter(self._plan)

class Experiment(object):
    evolution_parameters = ["run", "generation", "timestamp"]
    fitness_parameters = ["mse", "area", "energy", "delay", "depth", "gate_count"]
    chromosome_parameters = ["chromosome"]
    columns = evolution_parameters + fitness_parameters + chromosome_parameters

    def __init__(self, config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, parent: Optional[Self] = None) -> None:
        self.base_folder = config.path.parent
        self.set_paths(self.base_folder)
        self.parent: Self = None
        self.config = config 
        self._model = model
        self._cgp = cgp
        self.dtype = dtype
        self._to_number = int if self.dtype == torch.int8 else float
        self.model_acc = None
        self.model_loss = None
        self.reset()

    def get_name(self, depth: Optional[int] = None):
        parent = self.parent
        prev_parent = None
        while (parent is not None) and (depth is None or depth > 0):
            prev_parent, parent = parent, parent.parent
            depth = depth - 1 if depth is not None else None

        if prev_parent is None:
            return self.base_folder.stem

        raw_name = self.base_folder.relative_to(prev_parent)
        experiment_name = raw_name.stem
        final_name = raw_name.parent / experiment_name
        return os.path.normpath(os.path.normcase(final_name))


    def set_paths(self, root: Union[Path, str]):
        root = root if isinstance(root, Path) else Path(root)
        self.train_config = root / "train_cgp.config"
        self.eval_config = root / "eval_cgp.config"
        self.train_weights = root / "train.data"
        self.train_pbs = root / "train.pbs"
        self.eval_pbs = root / "eval.pbs"
        self.train_statistics = root / "train_statistics" / "statistics.{run}.csv"
        self.eval_statistics = root / "eval_statistics" / "statistics.csv"
        self.model_eval_statistics = root / "eval_statistics" / "model_statistics.csv"
        self.cached_model_attributes = root / "eval_statistics" / "cached_model_attributes.csv"
        self.result_configs = root / "cgp_configs" / "cgp.{run}.config"
        self.result_weights = root / "weights" / "weights.{run}.txt"
        self.gate_parameters_file = root / "gate_parameters.txt"
        self.train_stdout = root / "train_stdout.txt"
        self.train_stderr = root / "train_stderr.txt"
        self.eval_stdout = root / "eval_stdout.txt"
        self.eval_stderr = root / "eval_stderr.txt"

    def clean_train(self):
        shutil.rmtree(self.train_statistics.parent)
        shutil.rmtree(self.result_configs.parent)
        os.unlink(self.train_weights)
        os.unlink(self.train_stdout)
        os.unlink(self.train_stderr)
    
    def clean_eval(self):
        os.unlink(self.eval_stdout)
        os.unlink(self.eval_stderr)
        shutil.rmtree(self.eval_statistics.parent)
        shutil.rmtree(self.result_weights.parent)

    def clean_all(self):
        self.clean_train()
        self.clean_eval()

    def _clone(self, config: CGPConfiguration) -> Self:
        experiment = Experiment(config, self._model, self._cgp, self.dtype)
        experiment.model_acc = self.model_acc
        experiment.model_loss = self.model_loss
        experiment._cgp_prepared = self._cgp_prepared
        experiment._planner = self._planner.clone()
        experiment.parent = self.parent
        return experiment

    def setup_train_environment(self, config: CGPConfiguration = None, clean=False) -> Self:
        if clean:
            self.clean_train()

        experiment = self._clone(config or self.config.clone())
        experiment.result_folder.mkdir(exist_ok=False, parents=True)
        experiment.result_configs.parent.mkdir(exist_ok=False, parents=True)
        experiment.train_statistics.parent.mkdir(exist_ok=False, parents=True)
        experiment.eval_statistics.mkdir(exist_ok=True, parents=True)

        if not config.has_input_file():
            experiment._prepare_cgp(config)
            experiment._cgp.create_train_file(experiment.train_weights)
            config.set_input_file(experiment.train_weights)
        if not config.has_cgp_statistics_file():
            config.set_cgp_statistics_file(experiment.train_statistics)
        if not config.has_output_file():
            config.set_output_file(experiment.result_configs)
        if not config.has_output_file():
            config.set_output_file(experiment.result_configs)
        if not config.has_gate_parameters_file():
            config.set_gate_parameters_file(experiment.gate_parameters_file)
        if not config.has_stdout_file():
            config.set_stdout_file(experiment.train_stdout)
        if not config.has_stderr_file():
            config.set_stderr_file(experiment.train_stderr)

        return experiment

    def setup_isolated_train_environment(self, experiment_path: str, clean=False) -> Self:
        try:
            parent_path = self.parent.base_folder
            new_folder = Path(experiment_path) / (self.base_folder.relative_to(parent_path))
            new_folder.mkdir(exist_ok=False, parents=True)
            self.set_paths(new_folder)
            config = self.config.clone(self.train_config)
            if clean:
                if os.path.samefile(self.base_folder, experiment_path):
                    raise ValueError("cannot delete base experiment folder")
                shutil.rmtree(experiment_path)

            if not config.has_gate_parameters_file():
                raise ValueError("missing gate_parameters_file in the config")
            else:
                shutil.copyfile(config.get_gate_parameters_file(), self.gate_parameters_file)
                config.set_gate_parameters_file(self.gate_parameters_file)

            experiment = self.setup_train_environment(config)
            experiment.config.apply_extra_attributes()
            experiment.config.save()
            return experiment
        finally:
            self.set_paths(self.base_folder)

    def setup_eval_environment(self, clean: bool = False) -> Self:
        experiment = self._clone(self.config.clone(self.eval_config))
        if clean:
            experiment.clean_eval()

        experiment.eval_statistics.parent.mkdir(exist_ok=False, parents=True)
        experiment.result_weights.parent.mkdir(exist_ok=False, parents=True)
        experiment.config.set_input_file(self.result_configs)
        experiment.config.set_output_file(self.result_weights)
        experiment.config.set_cgp_statistics_file(self.eval_statistics)
        experiment.config.set_stdout_file(self.eval_stdout)
        experiment.config.set_stdout_file(self.eval_stderr)
        experiment.config.apply_extra_attributes()
        experiment.config.save()
        return experiment

    #  select=1:mem=16gb:scratch_local=10gb:ngpus=1:gpu_cap=cuda60:cuda_version=11.0 -q gpu -l walltime=4:00:00
    # select=1:ncpus={cpu}:mem={ram}:scratch_local={capacity}
    def generate_pbs(self):
        pass

    def setup_pbs_train_job(self,
                            machine: str,
                            time_limit: str,
                            template_pbs_file: str,
                            template_data: Dict[str, str],
                            experiments_folder: str = "experiments_folder",
                            results_folder: str = "results",
                            cgp_folder: str = "cgp_cpp_project"):
        template_data = {
            "machine": machine,
            "time_limit": time_limit,
            "job_name": self.get_name().replace("/", "_"),
            "server": os.environ.get("pbs_server"),
            "username": os.environ.get("pbs_username"),
            "workspace": f"/storage/{os.environ.get("pbs_server")}/home/{os.environ.get("pbs_username")}/cgp_workspace",
            "experiments_folder": experiments_folder,
            "results_folder": results_folder,
            "cgp_cpp_project": cgp_folder,
            "cgp_binary_src": "bin/cgp",
            "cgp_binary": "cgp",
            "cgp_command": "train",
            "cgp_config": self.config.path.name,
            "cgp_args": "todo",
            "experiment": self.get_name()
        }
# {os.path.normpath(os.path.normcase(self.base_folder))}
        with open(template_pbs_file, "r") as template_pbs_f, open(self.train_pbs, "w") as pbs_f:
            for line in template_pbs_f:
                changed = True
                while changed:
                    old_line = line
                    line = line.format(**template_data)
                    changed = line != old_line
                pbs_f.write(line)

    def reset(self):
        self._planner = CGPPinPlanner() 
        self._cgp_prepared = False

    def next_input_combination(self):
        self._planner.next_mapping()

    def add_filter_selector(
            self,
            sel: FilterSelector
        ):
        self._planner.add_mapping(sel)

    def _prepare_cgp(self, config: CGPConfiguration):
        if self._cgp_prepared: 
            return
        self._cgp.setup(config)
        try:
            self._model.eval()
            self._planner.finish_mapping()
            with torch.inference_mode():
                for combinational_plans in self._planner.get_plan():
                    for plan in combinational_plans:
                        weights = self._get_train_weights(plan.layer_name)
                        for in_selector in plan.inp:
                            self._cgp.add_inputs(weights[*in_selector])
                        for out_selector in plan.out:
                            self._cgp.add_outputs(weights[*out_selector])
                        self._cgp.next_train_item()
            self._cgp_prepared = True
        finally:
            self._model.train(True)

    def get_number_of_experiment_results(self) -> int:
        return len(os.listdir(self.result_configs.parent))

    def train(self, start_run: int = None, start_generation: int = None):
        config = self.config.clone()
        self._prepare_cgp(config)
        
        if start_run is not None:
            config.set_start_run(start_run)
        if start_generation is not None:
            config.set_start_generation(start_generation)

        print(self._cgp.get_train_cli(config))
        self._cgp.train()

    def evaluate(self):
        config = self.config.clone()
        self._prepare_cgp(config)
        self._cgp.evaluate()

    def get_model_statistics(self, run: int = None, top: int = 1):
        runs = [run] if run is not None else range(1, self.get_number_of_experiment_results()+1)
        for run in runs: 
            yield self.get_model_statistics(run=run, top=top)

    def get_model_statistics_from_file(self, file: str = None, run: int = 0, top: int = 1):
        new_model = self._inject_weights_from_file(file=file, run=run)
        after_acc, after_loss = new_model.evaluate(top=top)
        return after_acc, after_loss

    def evaluate_runs(self):
        self.evaluate(self.config)
        if self.model_acc is None or self.model_loss is None:
            self.model_acc, self.model_loss = self._model.evaluate()
        accuracies = []
        losses = []
        sources = []
        acc_delta = []
        loss_delta = []

        for run, acc, loss in enumerate(self.get_model_statistics(), start=1):
            accuracies.append(acc)
            losses.append(loss)
            sources.append(run)
            acc_delta.append(self.model_acc - acc)
            loss_delta.append(self.model_loss - loss)

        data = {"sources": sources, "accuracy": accuracies, "loss": losses, "accuracy_change": acc_delta, "loss_change": loss_delta}
        df = pd.DataFrame(data)
        df_model = pd.DataFrame({"accuracy": [self.model_acc], "loss": [self.model_loss]})
        print(df)
        df.to_csv(self.model_eval_statistics, index=False)
        df_model.to_csv(self.cached_model_attributes, index=False)
        return df

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
        with torch.inference_mode():
            offset = 0
            for weights, combination_plans in zip(weights_vector, self._planner.get_plan()):
                for plan in combination_plans:
                    bias = self._get_bias(plan.layer_name)
                    fp32_weights = self._get_reconstruction_weights(plan.layer_name)

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
                    self._set_weights_bias(getattr(model, plan.layer_name), fp32_weights, bias)
            return model

    def _inject_weights_from_file(self, file: str = None, run: int = 0):
        with torch.inference_mode():
            file = file or self._get_cgp_evaluate_file(run=run)
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
    return [(*selectors, *core)]

def conv2d_selector(layer_name: str, selectors, kernel_size, core_size):
    return FilterSelector(layer_name, conv2d_core(selectors, kernel_size, core_size), conv2d_outter(selectors, kernel_size, core_size))

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
