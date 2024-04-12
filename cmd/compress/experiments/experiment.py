import os
import torch
import pandas as pd
import shutil
from string import Template
from pathlib import Path
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from typing import Union, Self, Dict, Optional
from experiments.planner import CGPPinPlanner
from models.adapters.model_adapter import ModelAdapter
from models.selector import FilterSelector

class Experiment(object):
    evolution_parameters = ["run", "generation", "timestamp"]
    fitness_parameters = ["mse", "area", "energy", "delay", "depth", "gate_count"]
    chromosome_parameters = ["chromosome"]
    columns = evolution_parameters + fitness_parameters + chromosome_parameters

    def __init__(self, config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args,  dtype=torch.int8, parent: Optional[Self] = None) -> None:
        self.base_folder = config.path.parent
        self.temporary_base_folder = None
        self.set_paths(self.base_folder)
        self.parent: Self = parent
        self.config = config 
        self._model_adapter = model_adapter
        self._cgp = cgp
        self.dtype = dtype
        self._to_number = int if self.dtype == torch.int8 else float
        self.model_acc = None
        self.model_loss = None
        self.reset()

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
        self.train_config = root / "train_cgp.config"
        self.eval_config = root / "eval_cgp.config"
        self.train_weights = root / "train.data"
        self.train_pbs = root / "train.pbs.sh"
        self.eval_pbs = root / "eval.pbs.sh"
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
        experiment = Experiment(config, self._model_adapter.clone(), self._cgp, self.dtype)
        experiment.model_acc = self.model_acc
        experiment.model_loss = self.model_loss
        experiment._cgp_prepared = self._cgp_prepared
        experiment._planner = self._planner.clone()
        experiment.parent = self.parent
        return experiment

    def _handle_path(self, path: Path, relative: bool):
        return path if not relative else path.relative_to(self.temporary_base_folder or self.base_folder)

    def setup_train_environment(self, config: CGPConfiguration = None, clean=False, relative_paths: bool = False) -> Self:
        if clean:
            self.clean_train()

        experiment = self._clone(config or self.config.clone())
        exists_ok = self.get_number_of_experiment_results() == 0
        experiment.result_configs.parent.mkdir(exist_ok=exists_ok, parents=True)
        experiment.result_weights.parent.mkdir(exist_ok=exists_ok, parents=True)
        experiment.train_statistics.parent.mkdir(exist_ok=exists_ok, parents=True)
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

            if not config.has_gate_parameters_file():
                raise ValueError("missing gate_parameters_file in the config")
            else:
                shutil.copyfile(config.get_gate_parameters_file(), self.gate_parameters_file)
                config.set_gate_parameters_file(self.gate_parameters_file if not relative_paths else self.gate_parameters_file.name)

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

    def train(self, start_run: int = None, start_generation: int = None):
        config = self.config.clone()
        self._prepare_cgp(config)
        
        if start_run is not None:
            config.set_start_run(start_run)
        if start_generation is not None:
            config.set_start_generation(start_generation)

        print(self._cgp.get_cli_arguments(config))
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
        if self.get_number_of_experiment_results() != self.config.get_number_of_runs():
            self.evaluate()
        if self.model_acc is None or self.model_loss is None:
            self.model_acc, self.model_loss = self._model_adapter.evaluate()
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
    
    def _inject_weights_from_file(self, file: str = None, run: int = 0):
        with torch.inference_mode():
            file = file or str(self.result_configs).format(run=run)
            weights_vector = []
            with open(file) as f:
                for line, _ in zip(f.readlines(), range(self._cgp.config.get_dataset_size())):
                    segments = line.split(" ")

                    if "nan" in segments:
                        raise ValueError(f"CGP training failed for {file}; the file contains invalid weight")

                    weights = torch.Tensor([self._to_number(segment) for segment in segments if segment.strip() != ""])
                    weights_vector.append(weights)
            return self._model_adapter.inject_weights(weights_vector, self._planner.get_plan())    
