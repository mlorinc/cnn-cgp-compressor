import argparse
import csv
import os
import torch
from parse import parse
import shutil
from string import Template
from pathlib import Path
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from typing import Union, Self, Optional, List, Iterable
from experiments.planner import CGPPinPlanner
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from models.selector import FilterSelector
from circuit.loader import get_gate_parameters

class MissingChromosomeError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Experiment(object):
    evolution_parameters = ["run", "generation", "timestamp"]
    fitness_parameters = ["mse", "area", "energy", "delay", "depth", "gate_count"]
    chromosome_parameters = ["chromosome"]
    columns = evolution_parameters + fitness_parameters + chromosome_parameters
    train_cgp_name = "train_cgp.config"

    @staticmethod
    def get_train_pbs_argument_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--time-limit", required=True, help="Time limit for the PBS job")
        parser.add_argument("--template-pbs-file", required=True, help="Path to the template PBS file")
        parser.add_argument("--experiments-folder", default="experiments_folder", help="Experiments folder")
        parser.add_argument("--results-folder", default="results", help="Results folder")
        parser.add_argument("--cgp-folder", default="cgp_cpp_project", help="CGP folder")
        parser.add_argument("--cpu", type=int, default=None, help="Number of CPUs")
        parser.add_argument("--mem", default="2gb", help="Memory")
        parser.add_argument("--scratch-capacity", default="1gb", help="Scratch capacity")
        return parser

    def __init__(self, config: Union[CGPConfiguration, str, Path], model_adapter: ModelAdapter, cgp: CGP, args, dtype=torch.int8, parent: Optional[Self] = None) -> None:
        self.base_folder = config.path.parent if isinstance(config, CGPConfiguration) else Path(config)
        self.args = args
        if isinstance(config, CGPConfiguration):
            config.parse_arguments(args)

        self.temporary_base_folder = None
        self.set_paths(self.base_folder)
        self.parent: Self = parent
        self.config = config if isinstance(config, CGPConfiguration) else None
        self._model_adapter = model_adapter
        self._cgp = cgp
        self.dtype = dtype
        self._to_number = int if self.dtype == torch.int8 else float
        self.model_top_k = None
        self.model_loss = None
        self.reset()

    @classmethod
    def with_data_only(cls, path: Union[Path, str], model_name: str = None, model_path: str = None):
        path = Path(path)
        # model_adapter = BaseAdapter.load_base_model(model_name, model_path)
        model_adapter = None
        if path.name == Experiment.train_cgp_name:
            return cls(CGPConfiguration(path), model_adapter, model_path, None, {})
        else:
            return cls(CGPConfiguration(path / Experiment.train_cgp_name), model_adapter, None, {})

    def get_name(self, depth: Optional[int] = None):
        current_item = self
        names = []
        while (current_item is not None) and (depth is None or depth > 0):
            names.append(current_item.base_folder.name)
            current_item = current_item.parent
            depth = depth - 1 if depth is not None else None

        names = names[::-1]
        return "/".join(names)

    def set_paths(self, root: Union[Path, str]):
        root = root if isinstance(root, Path) else Path(root)
        self.train_config = root / self.train_cgp_name
        self.eval_config = root / "eval_cgp.config"
        self.train_weights = root / "train.data"
        self.train_pbs = root / "train.pbs.sh"
        self.eval_pbs = root / "eval.pbs.sh"
        self.train_statistics = root / "train_statistics" / "fitness" / "statistics.{run}.csv"
        self.eval_statistics = root / "eval_statistics" / "statistics.csv"
        self.model_eval_statistics = root / "eval_statistics" / "model_statistics.csv"
        self.cached_model_attributes = root / "eval_statistics" / "cached_model_attributes.csv"
        self.result_configs = root / "cgp_configs" / "cgp.{run}.config"
        self.result_weights = root / "weights" / "weights.{run}.txt"
        self.gate_parameters_file = root / "gate_parameters.txt"
        self.gate_parameters_csv_file = root / "gate_parameters.csv"
        self.train_stdout = root / "train_stdout.txt"
        self.train_stderr = root / "train_stderr.txt"
        self.eval_stdout = root / "eval_stdout.txt"
        self.eval_stderr = root / "eval_stderr.txt"
        self.learning_rate_file = root / "train_statistics" / "learning" / "learning_rate.{run}.csv"
        self.temporary_base_folder = root if root != self.base_folder else None

    def clean_train(self):
        shutil.rmtree(self.train_statistics.parent)
        shutil.rmtree(self.result_configs.parent)
        shutil.rmtree(self.result_weights.parent)
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
        experiment = Experiment(config, self._model_adapter.clone() if self._model_adapter else None, self._cgp, self.args, self.dtype)
        experiment.model_top_k = self.model_top_k
        experiment.model_loss = self.model_loss
        experiment._cgp_prepared = self._cgp_prepared
        experiment._planner = self._planner.clone()
        experiment.parent = self.parent
        experiment.args = self.args
        experiment.dtype = self.dtype
        return experiment

    def _handle_path(self, path: Path, relative: bool):
        return path if not relative else path.relative_to(self.temporary_base_folder or self.base_folder)

    def setup_resumed_train_environment(self, config: CGPConfiguration = None, relative_paths: bool = False, start_run=None, start_generation=None):
        experiment = self._clone(config or self.config.clone())
        if start_run is None and start_generation is None:
            last_run = self.get_number_of_experiment_results()
            if last_run == 0:
                return None            
            
            with open(str(self.train_statistics).format(run=start_run or last_run), "rb") as f:
                try:  # catch OSError in case of a one line file 
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b"\n":
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
                reader = csv.reader([last_line], delimiter=",")
                row = next(reader)
                
                start_run, start_generation, chromosome = start_run or row[0], start_generation or row[1], row[-1]
                if not chromosome:
                    raise MissingChromosomeError("missing chromosome")
                
                experiment.config.set_start_run(start_run)
                experiment.config.set_start_generation(start_generation)
                experiment.config.set_starting_solution(chromosome)
        else:
            experiment.config.set_start_run(start_run)
            experiment.config.set_start_generation(start_generation)
        return experiment

    def setup_train_environment(self, config: CGPConfiguration = None, clean=False, relative_paths: bool = False) -> Self:
        if clean:
            self.clean_train()

        experiment = self._clone(config or self.config.clone())
        exists_ok = not clean or self.get_number_of_experiment_results() == 0
        experiment.result_configs.parent.mkdir(exist_ok=exists_ok, parents=True)
        experiment.result_weights.parent.mkdir(exist_ok=exists_ok, parents=True)
        experiment.train_statistics.parent.mkdir(exist_ok=exists_ok, parents=True)
        experiment.learning_rate_file.parent.mkdir(exist_ok=exists_ok, parents=True)
        experiment.eval_statistics.parent.mkdir(exist_ok=True, parents=True)

        if not config.has_input_file():
            experiment._prepare_cgp(config)
            experiment._cgp.create_train_file(experiment.train_weights)
            config.set_input_file(self._handle_path(experiment.train_weights, relative_paths))
        if not config.has_cgp_statistics_file():
            config.set_cgp_statistics_file(self._handle_path(experiment.train_statistics, relative_paths))
        if not config.has_output_file():
            config.set_output_file(self._handle_path(experiment.result_configs, relative_paths))
        if not config.has_output_file():
            config.set_output_file(self._handle_path(experiment.result_configs, relative_paths))
        if not config.has_gate_parameters_file():
            config.set_gate_parameters_file(self._handle_path(experiment.gate_parameters_file, relative_paths))
        if not config.has_stdout_file():
            config.set_stdout_file(self._handle_path(experiment.train_stdout, relative_paths))
        if not config.has_stderr_file():
            config.set_stderr_file(self._handle_path(experiment.train_stderr, relative_paths))
        if not config.has_train_weights_file():
            config.set_train_weights_file(self._handle_path(self.result_weights, relative_paths))
        if not config.has_learning_rate_file():
            config.set_learning_rate_file(self._handle_path(self.learning_rate_file, relative_paths))            
        return experiment

    def setup_isolated_train_environment(self, experiment_path: str, clean=False, relative_paths: bool = False) -> Self:
        try:
            new_folder = Path(experiment_path) / (self.get_name())
            new_folder.mkdir(exist_ok=True, parents=True)
            self.set_paths(new_folder)
            print(f"creating new environment in {str(new_folder)}")
            config = self.config.clone(self.train_config)
            if clean:
                if os.path.samefile(self.base_folder, experiment_path):
                    raise ValueError("cannot delete base experiment folder")
                shutil.rmtree(experiment_path)

            # todo remove
            if True or not self.gate_parameters_file.exists():
                get_gate_parameters(
                    self.gate_parameters_csv_file,
                    self.gate_parameters_file,
                    grid_size=(config.get_row_count(), config.get_col_count()))
            config.set_gate_parameters_file(self._handle_path(self.gate_parameters_file, relative_paths))

            experiment = self.setup_train_environment(config, relative_paths=relative_paths)
            experiment.config.apply_extra_attributes()
            experiment.config.save()
            return experiment
        finally:
            self.set_paths(self.base_folder)

    def setup_eval_environment(self, clean: bool = False) -> Self:
        experiment = self._clone(self.config.clone(self.eval_config))
        if clean:
            experiment.clean_eval()

        experiment.eval_statistics.parent.mkdir(exist_ok=True, parents=True)
        experiment.result_weights.parent.mkdir(exist_ok=True, parents=True)
        experiment.config.set_input_file(self.result_configs)
        experiment.config.set_output_file(self.result_weights)
        experiment.config.set_cgp_statistics_file(self.eval_statistics)
        experiment.config.set_stdout_file(self.eval_stdout)
        experiment.config.set_stdout_file(self.eval_stderr)
        experiment.config.apply_extra_attributes()
        experiment.config.save()
        return experiment

    def setup_statistics_fix_environment(self) -> Self:
        experiment = self._clone(self.config.clone())
        experiment.config.set_input_file(self.train_weights)
        experiment.config.set_output_file(self.train_statistics.parent / (self.train_statistics.name + ".fixed"))
        experiment.config.set_cgp_statistics_file(self.train_statistics)
        return experiment        

    #  select=1:mem=16gb:scratch_local=10gb:ngpus=1:gpu_cap=cuda60:cuda_version=11.0 -q gpu -l walltime=4:00:00
    # select=1:ncpus={cpu}:mem={ram}:scratch_local={capacity}
    def setup_pbs_train_job(self,
                            time_limit: str,
                            template_pbs_file: str,
                            experiments_folder: str = "experiments_folder",
                            results_folder: str = "results",
                            cgp_folder: str = "cgp_cpp_project",
                            cpu=32,
                            mem="2gb",
                            scratch_capacity="1gb"):        
        args = self.config.to_args()
        job_name = self.get_name().replace("/", "_")
        template_data = {
            "machine": f"select=1:ncpus={cpu}:mem={mem}:scratch_local={scratch_capacity}",
            "time_limit": time_limit,
            "job_name": f"cgp_mlorinc_{job_name}",
            "server": os.environ.get("pbs_server"),
            "username": os.environ.get("pbs_username"),
            "workspace": "/storage/$server/home/$username/cgp_workspace",
            "experiments_folder": experiments_folder,
            "results_folder": results_folder,
            "cgp_cpp_project": cgp_folder,
            "cgp_binary_src": "bin/cgp",
            "cgp_binary": "cgp",
            "cgp_command": "train",
            "cgp_config": self.config.path.name,
            "cgp_args": " ".join([str(arg) for arg in args]),
            "experiment": self.get_name()
        }

        with open(template_pbs_file, "r") as template_pbs_f, open(self.train_pbs, "w", newline="\n") as pbs_f:
            copy_mode = False
            for line in template_pbs_f:
                if not copy_mode and line.strip() == "# PYTHON TEMPLATE END":
                    copy_mode = True
                    continue

                if copy_mode:
                    pbs_f.write(line)
                else:
                    # keep substituting until it is done
                    changed = True
                    while changed:
                        old_line = line
                        template = Template(line)
                        line = template.safe_substitute(template_data)
                        changed = line != old_line
                    pbs_f.write(line)
        print("saved pbs file to: " + str(self.train_pbs))

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
        original_training = self._model_adapter.model.training
        try:
            self._model_adapter.eval()
            self._planner.finish_mapping()
            with torch.inference_mode():
                for combinational_plans in self._planner.get_plan():
                    for plan in combinational_plans:
                        weights = self._model_adapter.get_train_weights(plan.layer_name)
                        for in_selector in plan.inp:
                            self._cgp.add_inputs(weights[*in_selector])
                        for out_selector in plan.out:
                            self._cgp.add_outputs(weights[*out_selector])
                        self._cgp.next_train_item()
            self._cgp_prepared = True
        finally:
            self._model_adapter.model.train(mode=original_training)

    def get_number_of_experiment_results(self) -> int:
        return len(os.listdir(self.result_configs.parent)) if self.result_configs.parent.exists() else 0

    def get_number_of_train_statistic_file(self) -> int:
        return len(os.listdir(self.train_statistics.parent)) if self.train_statistics.parent.exists() else 0

    def get_infered_weights_run_list(self) -> List[int]:
        runs = []
        for weight_file in os.listdir(self.result_weights.parent):
            runs.append(int(parse(self.result_weights.name, weight_file)["run"]))
        return runs

    def train_cgp(self, start_run: int = None, start_generation: int = None):
        config = self.config.clone()
        self._prepare_cgp(config)
        
        if start_run is not None:
            config.set_start_run(start_run)
        if start_generation is not None:
            config.set_start_generation(start_generation)

        self._cgp.train()

    def infer_missing_weights(self):
        config = self.config.clone()
        self._prepare_cgp(config)
        self._cgp.evaluate()

    def evaluate_missing_statistics(self):
        config = self.config.clone()
        self._cgp.setup(config)
        self._cgp.evaluate_all()

    def get_train_statistics(self, runs: Optional[Union[List[int], int]] = None) -> List[Path]:
        runs = runs if isinstance(runs, list) else [runs] if runs is not None else self.get_infered_weights_run_list()
        return [self.train_statistics.parent / (self.train_statistics.name.format(run=run)) for run in runs]

    def get_learn_rate_statistics(self, runs: Optional[Union[List[int], int]] = None) -> List[Path]:
        runs = runs if isinstance(runs, list) else [runs] if runs is not None else self.get_infered_weights_run_list()
        return [self.learning_rate_file.parent / (self.learning_rate_file.name.format(run=run)) for run in runs]

    def get_model_metrics(self,
                               clean=False,
                               only_path=True,
                               batch_size: int = 32,
                               max_batches: int = None,
                               top: Union[List[int], int] = [1, 5],
                               include_loss: bool = True,
                               show_top_k: int = 0):
        runs = self.get_infered_weights_run_list()
        if len(runs) != self.config.get_number_of_runs():
            self.infer_missing_weights()
        
        if clean or not self.model_eval_statistics.exists():
            with open(self.model_eval_statistics, "w") as file:
                csv_writer = csv.writer(file, lineterminator="\n", delimiter=",")
                headers = [f"top-{k}" for k in top] + ["loss"]         
                csv_writer.writerow(headers)
                metrics = []
                for run in runs:
                    weights, plans = self.get_weights(self.result_weights.parent / (self.result_weights.name.format(run=run)))
                    model = self._model_adapter.inject_weights(weights, plans, inline=False)
                    top_k, loss = model.evaluate(batch_size=batch_size, max_batches=max_batches, top=top, include_loss=include_loss, show_top_k=show_top_k)
                    values = list(top_k) + [loss]            
                    csv_writer.writerow(values)
                    if only_path:
                        metrics.append(values)
                return (headers, metrics) if not only_path else self.model_eval_statistics
        elif self.model_eval_statistics.exists():
            return self.model_eval_statistics
        else:
            raise NotImplementedError()

    def get_reference_model_metrics(self,
                                    file: Optional[Union[Path, str]] = None,
                                    cache: bool = True,
                                    batch_size: int = 32,
                                    max_batches: int = None,
                                    top: Union[List[int], int] = 1,
                                    include_loss: bool =True,
                                    show_top_k: int = 2):
        if self.model_top_k is not None and self.model_loss is not None:
            return self.model_top_k, self.model_loss
        elif self.cached_model_attributes.exists():
            with open(file or self.cached_model_attributes, "r") as f:
                csv_reader = csv.reader(f, lineterminator="\n", delimiter=",")
                headers = next(csv_reader)
                values = next(csv_reader)
                model_top_k = dict()
                loss = None
                for i, header in enumerate(headers):
                    parse_results = parse("top-{k}", header)
                    if parse_results is None and header != "loss":
                        raise ValueError(f"unknown header {header}")
                    elif header == "loss":
                        loss = values[i]
                    elif parse_results is not None and "k" in parse_results:
                        model_top_k[int(parse_results["k"])] = values[i]
                    else:
                        raise ValueError(f"unknown value to parse {header}")
                self.model_top_k = model_top_k
                self.model_loss = loss
                return self.model_top_k, self.model_loss
        else:
            self.model_top_k, self.model_loss = self._model_adapter.evaluate(batch_size=batch_size, max_batches=max_batches, top=top, include_loss=include_loss, show_top_k=show_top_k)            
            if cache:
                with open(self.cached_model_attributes, "w") as file:
                    csv_writer = csv.writer(file, lineterminator="\n", delimiter=",")
                    headers = [f"top-{k}" for k in top] + ["loss"]
                    csv_writer.writerow(headers)
                    csv_writer.writerow(list(self.model_top_k.values()) + [self.model_loss])
            return self.model_top_k, self.model_loss
    
    def get_weights(self, file: Optional[Union[Path, str, int]]):
        with torch.inference_mode():
            file = Path(file) if not isinstance(file, int) else str(self.result_configs).format(run=file)
            weights_vector = []
            with open(file) as f:
                for line in f:
                    segments = line.split(" ")

                    if "nan" in segments:
                        raise ValueError(f"CGP training failed for {file}; the file contains invalid weight")

                    weights = torch.Tensor([self._to_number(segment) for segment in segments if segment.strip() != ""])
                    weights_vector.append(weights)

                    if len(weights_vector) == self._cgp.config.get_dataset_size():
                        break;
            return weights_vector, self._planner.get_plan()
