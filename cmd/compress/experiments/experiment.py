# Copyright 2024 Marián Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# experiment.py: Baseline experiment implementation supporting single experiment without independent variables.
# The utility is also capable of generating PBS jobs.

import csv
import os
import shutil
import math
from string import Template
from pathlib import Path
from typing import Union, Self, Optional, List, Iterable
import torch
from parse import parse
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.quantization import tensor_iterator
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from models.selector import FilterSelectorCombinations
from circuit.loader import get_gate_parameters

class MissingChromosomeError(ValueError):
    """
    Custom exception to indicate a missing chromosome in the experiment.
    """    
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Experiment(object):
    """
    Experiment class to handle the setup, training, evaluation, and management of CGP experiments.
    """    
    train_cgp_name = "train_cgp.config"
    error_metric_macros = {
        "MSE": "-D__MEAN_SQUARED_ERROR_METRIC",
        "AE": "-D__ABSOLUTE_ERROR_METRIC",
        "SE": ""
    }
    error_metric_thresholds = {
        "MSE": lambda _, error=256, extra_error=0: int(error**2 + extra_error),
        "AE": lambda output_size, error=256, extra_error=0: int(error * output_size + extra_error),
        "SE": lambda output_size, error=256, extra_error=0: int(error**2 * output_size + extra_error)
    }

    def __init__(self, config: Union[CGPConfiguration, str, Path], model_adapter: ModelAdapter, cgp: CGP,
                 dtype=torch.int8, parent: Optional[Self] = None, start_run=None, number_of_runs=None, depth=None, allowed_mse_error=None, **kwargs) -> None:
        """
        Initialize the Experiment class.

        Args:
            config (Union[CGPConfiguration, str, Path]): Configuration for the CGP experiment.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype, optional): Data type for the experiment. Defaults to torch.int8.
            parent (Optional[Self], optional): Parent experiment. Defaults to None.
            start_run (optional): Starting run number. Defaults to None.
            number_of_runs (optional): Number of runs for the experiment. Defaults to None.
            depth (optional): Depth of the experiment. Defaults to None.
            allowed_mse_error (optional): Allowed MSE error for the experiment. Defaults to None.
            kwargs: Additional arguments for the experiment.
        """        
        self.base_folder = config.path.parent if isinstance(config, CGPConfiguration) else Path(config) if config is not None else None
        self.batched_parent = None
        self.args = dict(**kwargs)
        self._allowed_mse_error = allowed_mse_error
        self._start_run = start_run
        self._depth = depth
        if isinstance(config, CGPConfiguration):
            config.parse_arguments({**self.args, "start_run": start_run, "number_of_runs": number_of_runs})

        self.temporary_base_folder = None
        self.set_paths(self.base_folder)
        self.parent: Self = parent
        self.config = config if isinstance(config, CGPConfiguration) else None
        self._model_adapter = model_adapter
        self.dtype = dtype
        self._cgp = cgp if isinstance(cgp, CGP) else CGP(cgp, dtype=self.dtype)
        self._to_number = int if self.dtype == torch.int8 else float
        self.model_top_k = None
        self.model_loss = None
        self.name_fmt = None
        self._feature_maps_combinations = None
        self.error_threshold_function = self.error_metric_thresholds.get(self.args["e_fitness"].upper(), None)
        self.error_metric_macro = self.error_metric_macros.get(self.args["e_fitness"].upper(), None)
        self.reset()
        
        if self.error_threshold_function is None:
            raise ValueError(f"unknown error fitness metric {self.args['e_fitness']}")    

    @classmethod
    def with_data_only(cls, path: Union[Path, str], model_name: str = None, model_path: str = None, cgp: str = None):
        """
        Initialize an Experiment instance with only data.

        Args:
            path (Union[Path, str]): Path to the experiment.
            model_name (str, optional): Name of the model. Defaults to None.
            model_path (str, optional): Path to the model. Defaults to None.
            cgp (str, optional): Path to the CGP binary. Defaults to None.

        Returns:
            Experiment: An instance of the Experiment class.
        """        
        path = Path(path)
        model_adapter = BaseAdapter.load_base_model(model_name, model_path) if model_name and model_path else None
        if path.name == Experiment.train_cgp_name:
            return cls(CGPConfiguration(path), model_adapter, model_path, cgp, {})
        else:
            return cls(CGPConfiguration(path / Experiment.train_cgp_name), model_adapter, cgp, {})

    @classmethod
    def with_cli_arguments(cls, config: Union[CGPConfiguration, str, Path], model_adapter: ModelAdapter, cgp: CGP, args, prepare=True):
        """
        Initialize an Experiment instance with command line arguments.

        Args:
            config (Union[CGPConfiguration, str, Path]): Configuration for the CGP experiment.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            args: Command line arguments.
            prepare (bool, optional): Flag to prepare the experiment. Defaults to True.

        Returns:
            Experiment: An instance of the Experiment class.
        """        
        return cls(config, model_adapter, cgp, prepare=prepare, **args)

    def get_name(self, depth: Optional[int] = None):
        """
        Get the name of the experiment.

        Args:
            depth (Optional[int], optional): Depth of the name to retrieve. Defaults to None.

        Returns:
            str: Name of the experiment.
        """        
        current_item = self
        names = []
        while (current_item is not None) and (depth is None or depth > 0):
            names.append(current_item.base_folder.name)
            current_item = current_item.parent
            depth = depth - 1 if depth is not None else None

        names = names[::-1]
        return "/".join(names)

    def set_paths(self, root: Union[Path, str]):
        """
        Set the paths for the experiment files.

        Args:
            root (Union[Path, str]): Root path for the experiment.
        """        
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
        """
        Clean the training files for the experiment.
        """        
        shutil.rmtree(self.train_statistics.parent)
        shutil.rmtree(self.result_configs.parent)
        shutil.rmtree(self.result_weights.parent)
        os.unlink(self.train_weights)
        os.unlink(self.train_stdout)
        os.unlink(self.train_stderr)
    
    def clean_eval(self):
        """
        Clean the evaluation files for the experiment.
        """        
        os.unlink(self.eval_stdout)
        os.unlink(self.eval_stderr)
        shutil.rmtree(self.eval_statistics.parent)
        shutil.rmtree(self.result_weights.parent)

    def clean_all(self):
        """
        Clean all files for the experiment.
        """        
        self.clean_train()
        self.clean_eval()

    def _clone(self, config: CGPConfiguration) -> Self:
        """
        Clone the experiment instance.

        Args:
            config (CGPConfiguration): Configuration for the CGP experiment.

        Returns:
            Self: A cloned instance of the Experiment class.
        """        
        experiment = Experiment(
            config or self.config.clone(),
            self._model_adapter.clone() if self._model_adapter else None,
            self._cgp,
            dtype=self.dtype,
            start_run=self._start_run,
            depth=self._depth,
            allowed_mse_error=self._allowed_mse_error,
            **self.args
            )
        experiment.model_top_k = self.model_top_k
        experiment.model_loss = self.model_loss
        experiment._cgp_prepared = self._cgp_prepared
        experiment._feature_maps_combinations = self._feature_maps_combinations.clone()
        experiment.parent = self.parent
        return experiment

    def _handle_path(self, path: Path, relative: bool):
        """
        Handle the path for the experiment files.

        Args:
            path (Path): Path to handle.
            relative (bool): Flag to indicate if the path should be relative.

        Returns:
            Path: The handled path.
        """        
        return path if not relative else path.relative_to(self.temporary_base_folder or self.base_folder)

    def get_resumed_train_env(self, config: CGPConfiguration = None, relative_paths: bool = False, start_run=None, start_generation=None):
        """
        Get the resumed training environment for the experiment.

        Args:
            config (CGPConfiguration, optional): Configuration for the CGP experiment. Defaults to None.
            relative_paths (bool, optional): Flag to indicate if the paths should be relative. Defaults to False.
            start_run (optional): Starting run number. Defaults to None.
            start_generation (optional): Starting generation number. Defaults to None.

        Returns:
            Experiment: A resumed training environment for the experiment.
        """        
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

    def get_train_env(self, config: CGPConfiguration = None, clean=False, relative_paths: bool = False, reuse_weight_file=False) -> Self:
        """
        Get the training environment for the experiment.

        Args:
            config (CGPConfiguration, optional): Configuration for the CGP experiment. Defaults to None.
            clean (bool, optional): Flag to indicate if the training environment should be cleaned. Defaults to False.
            relative_paths (bool, optional): Flag to indicate if the paths should be relative. Defaults to False.
            reuse_weight_file (bool, optional): Flag to indicate if the weight file should be reused. Defaults to False.

        Returns:
            Experiment: A training environment for the experiment.
        """        
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
            if self.batched_parent is None and reuse_weight_file:
                assert isinstance(self.batched_parent, Experiment)
                shutil.copyfile(self.train_weights.parent.parent / self.batched_parent.get_name(depth=1) / self.train_weights.name, self.train_weights)
            else:
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
        if not config.has_mse_chromosome_logging_threshold() and self._allowed_mse_error is not None:
            allowed_mse_error = self._allowed_mse_error
            try:
                allowed_mse_error = int(allowed_mse_error)
            except:
                allowed_mse_error = float(allowed_mse_error)
            

            output_size = experiment.config.get_output_count() * experiment.config.get_dataset_size()
            if isinstance(allowed_mse_error, int) or isinstance(allowed_mse_error, float):
                config.set_mse_chromosome_logging_threshold(int(self.error_threshold_function(output_size, error=allowed_mse_error)))
            elif allowed_mse_error is not None:
                raise TypeError("allowed-mse-error must be either int or float: " + str(type(allowed_mse_error)))
        if not config.has_start_run() and self._start_run:
            config.set_start_run(self._start_run)
        return experiment

    def get_isolated_train_env(self, experiment_path: str, clean=False, relative_paths: bool = False) -> Self:
        """
        Get an isolated training environment for the experiment.

        Args:
            experiment_path (str): Path to the isolated training environment.
            clean (bool, optional): Flag to indicate if the training environment should be cleaned. Defaults to False.
            relative_paths (bool, optional): Flag to indicate if the paths should be relative. Defaults to False.

        Returns:
            Experiment: An isolated training environment for the experiment.
        """        
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
                _, self.energy_series, self.delay_series = get_gate_parameters(
                    self.gate_parameters_csv_file,
                    self.gate_parameters_file,
                    grid_size=(config.get_row_count(), config.get_col_count()))
                
            config.set_gate_parameters_file(self._handle_path(self.gate_parameters_file, relative_paths))

            experiment = self.get_train_env(config, relative_paths=relative_paths, reuse_weight_file=False)
            experiment.config.apply_extra_attributes()
            experiment.config.save()
            return experiment
        finally:
            self.set_paths(self.base_folder)

    def get_result_eval_env(self, clean: bool = False) -> Self:
        """
        Get the result evaluation environment for the experiment.

        Args:
            clean (bool, optional): Flag to indicate if the evaluation environment should be cleaned. Defaults to False.

        Returns:
            Experiment: A result evaluation environment for the experiment.
        """        
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
        return experiment

    def get_statistics_fix_env(self, inline=False) -> Self:
        """
        Get the statistics fix environment for the experiment. Used if there was an issue with
        evaluation to get the assesed again.

        Args:
            inline (bool, optional): Flag to indicate if the environment should be inline. Defaults to False.

        Returns:
            Experiment: A statistics fix environment for the experiment.
        """        
        experiment = self._clone(self.config.clone()) if not inline else self
        experiment.config.set_input_file(self.train_weights)
        experiment.config.set_output_file(self.train_statistics.parent / (self.train_statistics.name + ".fixed"))
        experiment.config.set_cgp_statistics_file(self.train_statistics)
        return experiment        

    def get_model_metrics_env(self, output_file: Optional[Union[Path, str]] = None, inline=False) -> Self:
        """
        Get the model metrics environment for the experiment.

        Args:
            output_file (Optional[Union[Path, str]], optional): Path to the output file. Defaults to None.
            inline (bool, optional): Flag to indicate if the environment should be inline. Defaults to False.

        Returns:
            Experiment: A model metrics environment for the experiment.
        """        
        assert output_file is not None
        experiment = self._clone(self.config.clone()) if not inline else self
        experiment.config.set_input_file(self.train_weights)
        experiment.config.set_output_file(output_file or self.train_statistics.parent / (self.train_statistics.name + ".eval"))
        experiment.config.set_train_weights_file(output_file)
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
        """
        Set up the PBS training job for the experiment.

        Args:
            time_limit (str): Time limit for the PBS job.
            template_pbs_file (str): Path to the PBS template file.
            experiments_folder (str, optional): Path to the experiments folder on remote. Defaults to "experiments_folder".
            results_folder (str, optional): Path to the results folder on remote. Defaults to "results".
            cgp_folder (str, optional): Path to the CGP folder on remote. Defaults to "cgp_cpp_project".
            cpu (int, optional): Number of CPUs for the PBS job. Defaults to 32.
            mem (str, optional): Memory for the PBS job. Defaults to "2gb".
            scratch_capacity (str, optional): Scratch capacity for the PBS job. Defaults to "1gb".
        """        
        unsigned_types = {
            8: "uint8_t",
            16: "uint16_t",
            32: "uint32_t",
            64: "uint64_t",
        }
        
        error_bits = math.ceil(math.log2(self.error_threshold_function(self.config.get_output_count(), extra_error=1)))
        error_type = next(t for bit, t in unsigned_types.items() if error_bits <= bit)
        
        args = self.config.to_args()
        job_name = self.get_name().replace("/", "_")
        cxx_flags = ["-D_DISABLE_ROW_COL_STATS"]
        
        if not self._depth:
            cxx_flags.append("-D_DEPTH_DISABLED")
        
        if self.config.get_function_output_arity() == 1:
            cxx_flags.append("-D__SINGLE_OUTPUT_ARITY")  
        
        if self.args.get("multiplex", False) and self.config.get_dataset_size() == 1:
            cxx_flags.append("-D__SINGLE_MULTIPLEX")        
            
        if self.args.get("multiplex", False) and self.config.get_dataset_size() > 1:
            cxx_flags.append("-D__MULTI_MULTIPLEX")                 
        
        if self.error_metric_macro.strip() != "":
            cxx_flags.append(self.error_metric_macro)
        
        template_data = {
            "machine": f"select=1:ncpus={cpu}:ompthreads={cpu}:mem={mem}:scratch_local={scratch_capacity}",
            "time_limit": time_limit,
            "job_name": f"cgp_{job_name}",
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
            "experiment": self.get_name(),
            "error_t": error_type,
            "cflags": " ".join([str(arg) for arg in cxx_flags])
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

    def decode_configuration(self):
        """
        Decode the configuration for the experiment.

        Returns:
            dict: Decoded configuration.
        """        
        if self.name_fmt is None:
            raise ValueError(f"decoding not supported for {self.__class__.__name__}")
        
        return parse(self.name_fmt, self.get_name())

    def reset(self):
        """
        Reset the CGP preparation flag.
        """        
        self._cgp_prepared = False

    def get_input_combinations(self) -> FilterSelectorCombinations:
        """
        Get the input combinations for the experiment.

        Returns:
            FilterSelectorCombinations: Input combinations for the experiment.
        """        
        if self._feature_maps_combinations:
            return self._feature_maps_combinations
        raise NotImplementedError() 

    def set_feature_maps_combinations(self, combinations: FilterSelectorCombinations):
        """
        Set the feature maps combinations for the experiment.

        Args:
            combinations (FilterSelectorCombinations): Feature maps combinations.
        """        
        self._feature_maps_combinations = combinations

    def _prepare_cgp(self, config: CGPConfiguration):
        """
        Prepare the CGP for the experiment.

        Args:
            config (CGPConfiguration): Configuration for the CGP experiment.
        """        
        if self._cgp_prepared: 
            return
        self._cgp.setup(config)
        original_training = self._model_adapter.model.training
        try:
            self._model_adapter.eval()
            with torch.inference_mode():
                for combination in self.get_input_combinations().get_combinations():
                    for selector in combination.get_selectors():
                        weights = self._model_adapter.get_train_weights(selector.selector)
                        for w, _, _ in tensor_iterator(weights, selector.inp):
                            self._cgp.add_inputs(w)
                        for w, _, _ in tensor_iterator(weights, selector.out):
                            self._cgp.add_outputs(w)
                    self._cgp.next_train_item()
            self._cgp_prepared = True
        finally:
            self._model_adapter.model.train(mode=original_training)

    def get_number_of_experiment_results(self) -> int:
        """
        Get the number of experiment results.

        Returns:
            int: Number of experiment results.
        """        
        return len(os.listdir(self.result_configs.parent)) if self.result_configs.parent.exists() else 0

    def get_experiment_results_run_list(self) -> List[int]:
        """
        Get the list of experiment result runs.

        Returns:
            List[int]: List of experiment result runs.
        """        
        def f(x, extension=""):
            result = parse(self.result_configs.name + extension, x)
            return result["run"] if result else None
            
        runs = [int(f(file)) for file in os.listdir(self.result_configs.parent) if f(file)]       
        return runs or [int(f(file, extension=".zip")) for file in os.listdir(self.result_configs.parent) if f(file, extension=".zip")]

    def get_number_of_train_statistic_file(self, fmt: str = None) -> int:
        """
        Get the number of train statistic files.

        Args:
            fmt (str, optional): Format of the train statistic files. Defaults to None.

        Returns:
            int: Number of train statistic files.
        """        
        fmt = fmt or self.train_statistics.name
        def f(x):
            result = parse(fmt, x)
            return result["run"] if result else None        
        
        return [int(f(file)) for file in os.listdir(self.train_statistics.parent) if f(file)]

    def get_infered_weights_run_list(self) -> List[int]:
        """
        Get the list of inferred weights runs.

        Returns:
            List[int]: List of inferred weights runs.
        """        
        runs = []
        for weight_file in os.listdir(self.result_weights.parent):
            runs.append(int(parse(self.result_weights.name, weight_file)["run"]))
        return runs

    def train_cgp(self, start_run: int = None, start_generation: int = None):
        """
        Train the CGP for the experiment.

        Args:
            start_run (int, optional): Starting run number. Defaults to None.
            start_generation (int, optional): Starting generation number. Defaults to None.
        """        
        config = self.config.clone()
        self._prepare_cgp(config)
        
        if start_run is not None:
            config.set_start_run(start_run)
        if start_generation is not None:
            config.set_start_generation(start_generation)

        self._cgp.train()

    def infer_missing_weights(self):
        """
        Infer the missing weights for the experiment.
        """        
        config = self.config.clone()
        self._prepare_cgp(config)
        self._cgp.evaluate()

    def evaluate_chromosome_in_statistics(self, statistics: Union[Path, str], output_statistics: Union[Path, str], output_weights: Union[Path, str], mse_threshold=None):
        """
        Evaluate the chromosome in the statistics file.

        Args:
            statistics (Union[Path, str]): Path to the statistics file.
            output_statistics (Union[Path, str]): Path to the output statistics file.
            output_weights (Union[Path, str]): Path to the output weights file.
            mse_threshold (optional): MSE threshold. Defaults to None.
        """        
        assert statistics != output_statistics
        config = self.config.clone()
        config.set_input_file(self.train_weights)
        config.set_cgp_statistics_file(statistics)
        config.set_output_file(output_statistics)
        config.set_train_weights_file(output_weights)
        
        if mse_threshold:
            config.set_mse_chromosome_logging_threshold(mse_threshold)
        
        self._cgp.setup(config)
        self._cgp.evaluate_all()

    def evaluate_chromosomes(self, chromosomes_file: Union[Path, str], output_statistics: Union[Path, str], output_weights: Union[Path, str], gate_statistics_file: Union[Path, str]):
        """
        Evaluate the chromosomes in the file.

        Args:
            chromosomes_file (Union[Path, str]): Path to the chromosomes file.
            output_statistics (Union[Path, str]): Path to the output statistics file.
            output_weights (Union[Path, str]): Path to the output weights file.
            gate_statistics_file (Union[Path, str]): Path to the gate statistics file.
        """        
        config = self.config.clone()
        config.set_input_file(self.train_weights)
        config.set_cgp_statistics_file(chromosomes_file)
        config.set_output_file(output_statistics)
        config.set_train_weights_file(output_weights)
        config.set_gate_parameters_file(self.gate_parameters_file)
        self._cgp.setup(config)
        self._cgp.evaluate_chromosomes(gate_statistics_file)
        
    def evaluate_chromosome(self, chromosome: Union[Path, str], output_file: Union[Path, str] = "-", weights_file: Union[Path, str] = "-"):
        """
        Evaluate a single chromosome.

        Args:
            chromosome (Union[Path, str]): Path to the chromosome file.
            output_file (Union[Path, str], optional): Path to the output file. Defaults to "-".
            weights_file (Union[Path, str], optional): Path to the weights file. Defaults to "-".

        Returns:
            List[str]: Weights for the chromosome.
        """        
        config = self.config.clone()
        if weights_file == "-":
            weights_file = ".chromosome.temp"        
        config.set_input_file(self.train_weights)
        config.set_output_file(output_file)
        config.set_train_weights_file(weights_file)
        config.set_gate_parameters_file(self.gate_parameters_file)
        
        self._cgp.setup(config)
        self._cgp.evaluate_chromosome(chromosome)        

        if weights_file == ".chromosome.temp":
            weights = None
            with open(weights_file, "r") as f:
                weights = f.readlines()
            os.remove(weights_file)
            return weights
        return None
                

    def get_train_statistics(self, runs: Optional[Union[List[int], int]] = None, extension: Optional[str] = "", fmt: Optional[str] = None) -> List[Path]:
        """
        Get the training statistics files.

        Args:
            runs (Optional[Union[List[int], int]], optional): List of runs. Defaults to None.
            extension (Optional[str], optional): File extension. Defaults to "".
            fmt (Optional[str], optional): Format of the files. Defaults to None.

        Returns:
            List[Path]: List of training statistics files.
        """        
        if extension and not extension.startswith("."): extension = "." + extension
        
        runs = runs if isinstance(runs, list) else [runs] if runs is not None else self.get_experiment_results_run_list()
        return [self.train_statistics.parent / ((fmt or self.train_statistics.name).format(run=run) + extension) for run in runs]

    def get_learn_rate_statistics(self, runs: Optional[Union[List[int], int]] = None) -> List[Path]:
        """
        Get the learning rate statistics files.

        Args:
            runs (Optional[Union[List[int], int]], optional): List of runs. Defaults to None.

        Returns:
            List[Path]: List of learning rate statistics files.
        """        
        runs = runs if isinstance(runs, list) else [runs] if runs is not None else self.get_infered_weights_run_list()
        return [self.learning_rate_file.parent / (self.learning_rate_file.name.format(run=run)) for run in runs]

    def get_model_metrics(self,
                               weight_files: Optional[Union[Path, str]] = None,
                               output_file: Optional[Union[Path, str]] = None,
                               append=True,
                               clean=False,
                               batch_size: int = 32,
                               max_batches: int = None,
                               top: Union[List[int], int] = [1, 5],
                               include_loss: bool = True,
                               show_top_k: int = 0):
        """
        Get the model metrics for the experiment.

        Args:
            weight_files (Optional[Union[Path, str]], optional): List of weight files. Defaults to None.
            output_file (Optional[Union[Path, str]], optional): Path to the output file. Defaults to None.
            append (bool, optional): Flag to indicate if the output file should be appended. Defaults to True.
            clean (bool, optional): Flag to indicate if the output file should be cleaned. Defaults to False.
            batch_size (int, optional): Batch size for evaluation. Defaults to 32.
            max_batches (int, optional): Maximum number of batches for evaluation. Defaults to None.
            top (Union[List[int], int], optional): List of top-k accuracies to compute. Defaults to [1, 5].
            include_loss (bool, optional): Flag to indicate if loss should be included. Defaults to True.
            show_top_k (int, optional): Number of top-k accuracies to display. Defaults to 0.

        Returns:
            Path: Path to the output file.
        """        
        weight_files = weight_files or [self.result_weights.parent / (self.result_weights.name.format(run=run)) for run in self.get_infered_weights_run_list()]
        output_file = Path(output_file) or self.model_eval_statistics

        if clean or not output_file.exists():
            with open(output_file, "w" if not append else "a") as file:
                csv_writer = csv.writer(file, lineterminator="\n", delimiter=",")
                if not append:
                    headers = [f"top-{k}" for k in top] + ["loss"]         
                    csv_writer.writerow(headers)
                for file in weight_files:
                    weights, plans = self.get_weights(file)
                    model = self._model_adapter.inject_weights(weights, plans, inline=False)
                    top_k, loss = model.evaluate(batch_size=batch_size, max_batches=max_batches, top=top, include_loss=include_loss, show_top_k=show_top_k)
                    values = list(top_k) + [loss]            
                    csv_writer.writerow(values)
                return output_file
        elif output_file.exists():
            return output_file
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
        """
        Get the reference model metrics for the experiment.

        Args:
            file (Optional[Union[Path, str]], optional): Path to the cached model attributes file. Defaults to None.
            cache (bool, optional): Flag to indicate if the metrics should be cached. Defaults to True.
            batch_size (int, optional): Batch size for evaluation. Defaults to 32.
            max_batches (int, optional): Maximum number of batches for evaluation. Defaults to None.
            top (Union[List[int], int], optional): List of top-k accuracies to compute. Defaults to 1.
            include_loss (bool, optional): Flag to indicate if loss should be included. Defaults to True.
            show_top_k (int, optional): Number of top-k accuracies to display. Defaults to 2.

        Returns:
            Tuple[dict, float]: Dictionary of top-k accuracies and loss.
        """        
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
        """
        Get the weights for the experiment.

        Args:
            file (Optional[Union[Path, str, int]], optional): Path to the weight file. Defaults to None.

        Returns:
            Tuple[List[torch.Tensor], FilterSelectorCombinations]: List of weights and feature maps combinations.
        """        
        with torch.inference_mode():
            file = Path(file) if not isinstance(file, int) else str(self.result_weights).format(run=file)
            with open(file) as f:
                return self.parse_weights(f)

    def parse_weights(self, weights: Union[List[str], str]):
        """
        Parse the weights for the experiment.

        Args:
            weights (Union[List[str], str]): List of weight strings or weight file content.

        Returns:
            Tuple[List[torch.Tensor], FilterSelectorCombinations]: List of weights and feature maps combinations.
        """        
        with torch.inference_mode():
            weights = weights if isinstance(weights, Iterable) else [weights]
            weights_vector = []
            for line in weights:
                segments = line.strip().split(" ")

                if "nan" in segments:
                    raise ValueError(f"CGP training failed for {line}; the file contains invalid weight")

                weights = torch.Tensor([self._to_number(segment) for segment in segments if segment.strip() != ""])
                weights_vector.append(weights)

                if len(weights_vector) == self.config.get_dataset_size():
                    break;
            return weights_vector, self.get_input_combinations()
