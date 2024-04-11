from pathlib import Path
from typing import TextIO, Optional, Union
import contextlib
import subprocess
import os

class CGPConfiguration:
    ignored_arguments = set(["stdout", "stderr"])
    def __init__(self, config_file: Optional[Union[Path, str]] = None):
        self._extra_attributes = {}
        self._attributes = {}
        self.path = Path(config_file)
        if config_file:
            self.load(config_file)

    def clone(self, new_config_file: str = None):
        cloned_instance = CGPConfiguration()
        cloned_instance.path = new_config_file or self.path
        cloned_instance._attributes = self._attributes.copy()
        cloned_instance._extra_attributes = self._extra_attributes.copy()
        return cloned_instance

    def load(self, config_file: str = None):
        if config_file is None and self.path is None:
            raise ValueError(
                "either config file must be passed to the load function or the class constructor must have been provided a configuration file as argument"
            )
        with open(config_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines
                if line:
                    colon_index = line.index(":")
                    key, value = line[:colon_index], line[colon_index+1:]
                    self._attributes[key.strip()] = self._parse_value(value.strip())

    def save(self, config_file: str = None):
        if config_file is None and self.path is None:
            raise ValueError(
                "either config file must be passed to the save function or the class constructor must have been provided a configuration file as argument"
            )

        with open(config_file or self.path, "w") as f:
            for key, value in self._attributes.items():
                f.write(f"{key}: {value}\n")
        self.path = config_file or self.path

    def __contains__(self, key):
        return key in self._attributes

    def _parse_value(self, value_str: str):
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str

    def remove_extra_attributes(self):
        self._extra_attributes = {}

    def apply_extra_attributes(self):
        self._attributes = {**self._attributes, **self._extra_attributes}
        self._extra_attributes = {}

    def should_resume_evolution(self):
        resumed_run = (self.has_start_run() and self.get_start_run() != 0)
        resumed_generation = (self.has_start_generation() and self.get_start_generation() != 0)
        return resumed_run or resumed_generation

    @contextlib.contextmanager
    def open_stdout(self, mode="w"):
        f_handle: TextIO = None 
        try:
            file = self.get_stdout_file() if self.has_stdout_file() else "-"

            if file != "-":
                f_handle = open(file, mode)
                yield f_handle
            else:
                yield subprocess.PIPE
        finally:
            if f_handle is not None:
                f_handle.close()

    @contextlib.contextmanager
    def open_stderr(self, mode="w"):
        f_handle: TextIO = None 
        try:
            file = self.get_stderr_file() if self.has_stderr_file() else "-"

            if file != "-":
                f_handle = open(file, mode)
                yield f_handle
            else:
                yield subprocess.PIPE
        finally:
            if f_handle is not None:
                f_handle.close()

    def get_attribute(self, name):
        return self._attributes.get(name, None) or self._extra_attributes.get(name)

    def get_gate_parameters_file(self):
        return self.get_attribute("gate_parameters_file")

    def get_stderr_file(self):
        return self.get_attribute("stderr")

    def get_stdout_file(self):
        return self.get_attribute("stdout")

    def get_start_generation(self):
        return self.get_attribute("start_generation")

    def get_start_run(self):
        return self.get_attribute("start_run")

    def get_energy_early_stop(self):
        return self.get_attribute("energy_early_stop")

    def get_mse_early_stop(self):
        return self.get_attribute("mse_early_stop")

    def get_patience(self):
        return self.get_attribute("patience")

    def get_starting_solution(self):
        return self.get_attribute("starting_solution")

    def get_mse_threshold(self):
        return self.get_attribute("mse_threshold")

    def get_dataset_size(self):
        return self.get_attribute("dataset_size")

    def get_col_count(self):
        return self.get_attribute("col_count")

    def get_row_count(self):
        return self.get_attribute("row_count")

    def get_number_of_runs(self):
        return self.get_attribute("number_of_runs")

    def get_look_back_parameter(self):
        return self.get_attribute("look_back_parameter")

    def get_mutation_max(self):
        return self.get_attribute("mutation_max")

    def get_function_count(self):
        return self.get_attribute("function_count")

    def get_function_input_arity(self):
        return self.get_attribute("function_input_arity")

    def get_function_output_arity(self):
        return self.get_attribute("function_output_arity")

    def get_input_count(self):
        return self.get_attribute("input_count")

    def get_output_count(self):
        return self.get_attribute("output_count")

    def get_population_max(self):
        return self.get_attribute("population_max")

    def get_generation_count(self):
        return self.get_attribute("generation_count")

    def get_periodic_log_frequency(self):
        return self.get_attribute("periodic_log_frequency")

    def get_input_file(self):
        return self.get_attribute("input_file")

    def get_output_file(self):
        return self.get_attribute("output_file")

    def get_cgp_statistics_file(self):
        return self.get_attribute("cgp_statistics_file")

    def set_attribute(self, attribute, value):
        self._extra_attributes[attribute] = value

    def set_gate_parameters_file(self, value):
        self.set_attribute("gate_parameters_file", value)

    def set_start_generation(self, value):
        self.set_attribute("start_generation", value)

    def set_stderr_file(self, value):
        self.set_attribute("stderr", value)

    def set_stdout_file(self, value):
        self.set_attribute("stdout", value)

    def set_start_run(self, value):
        self.set_attribute("start_run", value)

    def set_energy_early_stop(self, value):
        self.set_attribute("energy_early_stop", value)

    def set_mse_early_stop(self, value):
        self.set_attribute("mse_early_stop", value)

    def set_patience(self, value):
        self.set_attribute("patience", value)

    def set_starting_solution(self, value):
        self.set_attribute("starting_solution", value)

    def set_mse_threshold(self, value):
        self.set_attribute("mse_threshold", value)

    def set_dataset_size(self, value):
        self.set_attribute("dataset_size", value)

    def set_col_count(self, value):
        self.set_attribute("col_count", value)

    def set_row_count(self, value):
        self.set_attribute("row_count", value)

    def set_number_of_runs(self, value):
        self.set_attribute("number_of_runs", value)

    def set_look_back_parameter(self, value):
        self.set_attribute("look_back_parameter", value)

    def set_mutation_max(self, value):
        self.set_attribute("mutation_max", value)

    def set_function_count(self, value):
        self.set_attribute("function_count", value)

    def set_function_input_arity(self, value):
        self.set_attribute("function_input_arity", value)

    def set_function_output_arity(self, value):
        self.set_attribute("function_output_arity", value)

    def set_input_count(self, value):
        self.set_attribute("input_count", value)

    def set_output_count(self, value):
        self.set_attribute("output_count", value)

    def set_population_max(self, value):
        self.set_attribute("population_max", value)

    def set_generation_count(self, value):
        self.set_attribute("generation_count", value)

    def set_periodic_log_frequency(self, value):
        self.set_attribute("periodic_log_frequency", value)

    def set_input_file(self, value):
        self.set_attribute("input_file", value)

    def set_output_file(self, value):
        self.set_attribute("output_file", value)

    def set_cgp_statistics_file(self, value):
        self.set_attribute("cgp_statistics_file", value)

    def delete_attribute(self, name):
        if name in self._attributes:
            del self._attributes[name]
        if name in self._extra_attributes:
            del self._extra_attributes[name]

    def delete_gate_parameters_file(self):
        self.delete_attribute("gate_parameters_file")

    def delete_stderr_file(self):
        self.delete_attribute("stderr")

    def delete_stdout_file(self):
        self.delete_attribute("stdout")

    def delete_start_generation(self):
        self.delete_attribute("start_generation")

    def delete_start_run(self):
        self.delete_attribute("start_run")

    def delete_energy_early_stop(self):
        self.delete_attribute("energy_early_stop")

    def delete_mse_early_stop(self):
        self.delete_attribute("mse_early_stop")

    def delete_patience(self):
        self.delete_attribute("patience")

    def delete_starting_solution(self):
        self.delete_attribute("starting_solution")

    def delete_mse_threshold(self):
        self.delete_attribute("mse_threshold")

    def delete_dataset_size(self):
        self.delete_attribute("dataset_size")

    def delete_col_count(self):
        self.delete_attribute("col_count")

    def delete_row_count(self):
        self.delete_attribute("row_count")

    def delete_number_of_runs(self):
        self.delete_attribute("number_of_runs")

    def delete_look_back_parameter(self):
        self.delete_attribute("look_back_parameter")

    def delete_mutation_max(self):
        self.delete_attribute("mutation_max")

    def delete_function_count(self):
        self.delete_attribute("function_count")

    def delete_function_input_arity(self):
        self.delete_attribute("function_input_arity")

    def delete_function_output_arity(self):
        self.delete_attribute("function_output_arity")

    def delete_input_count(self):
        self.delete_attribute("input_count")

    def delete_output_count(self):
        self.delete_attribute("output_count")

    def delete_population_max(self):
        self.delete_attribute("population_max")

    def delete_generation_count(self):
        self.delete_attribute("generation_count")

    def delete_periodic_log_frequency(self):
        self.delete_attribute("periodic_log_frequency")

    def delete_input_file(self):
        self.delete_attribute("input_file")

    def delete_output_file(self):
        self.delete_attribute("output_file")

    def delete_cgp_statistics_file(self):
        self.delete_attribute("cgp_statistics_file")

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in self._attributes.items()]
        return "\n".join(attributes)

    def has_attribute(self, name):
        return name in self._attributes or name in self._extra_attributes

    def has_gate_parameters_file(self):
        return self.has_attribute("gate_parameters_file")

    def has_stderr_file(self):
        return self.has_attribute("stderr")

    def has_stdout_file(self):
        return self.has_attribute("stdout")

    def has_start_generation(self):
        return self.has_attribute("start_generation")

    def has_start_run(self):
        return self.has_attribute("start_run")

    def has_energy_early_stop(self):
        return self.has_attribute("energy_early_stop")

    def has_mse_early_stop(self):
        return self.has_attribute("mse_early_stop")

    def has_patience(self):
        return self.has_attribute("patience")

    def has_starting_solution(self):
        return self.has_attribute("starting_solution")

    def has_mse_threshold(self):
        return self.has_attribute("mse_threshold")

    def has_dataset_size(self):
        return self.has_attribute("dataset_size")

    def has_col_count(self):
        return self.has_attribute("col_count")

    def has_row_count(self):
        return self.has_attribute("row_count")

    def has_number_of_runs(self):
        return self.has_attribute("number_of_runs")

    def has_look_back_parameter(self):
        return self.has_attribute("look_back_parameter")

    def has_mutation_max(self):
        return self.has_attribute("mutation_max")

    def has_function_count(self):
        return self.has_attribute("function_count")

    def has_function_input_arity(self):
        return self.has_attribute("function_input_arity")

    def has_function_output_arity(self):
        return self.has_attribute("function_output_arity")

    def has_input_count(self):
        return self.has_attribute("input_count")

    def has_output_count(self):
        return self.has_attribute("output_count")

    def has_population_max(self):
        return self.has_attribute("population_max")

    def has_generation_count(self):
        return self.has_attribute("generation_count")

    def has_periodic_log_frequency(self):
        return self.has_attribute("periodic_log_frequency")

    def has_input_file(self):
        return self.has_attribute("input_file")

    def has_output_file(self):
        return self.has_attribute("output_file")

    def has_cgp_statistics_file(self):
        return self.has_attribute("cgp_statistics_file")

    def to_args(self):
        arguments = []
        for k, v in self._extra_attributes.items():
            if k in CGPConfiguration.ignored_arguments:
                continue

            if k.endswith("_file"):
                v = os.path.normcase(os.path.normpath(str(v)))

            key = k.replace("_", "-")
            arguments.append(f"--{key}")
            arguments.append(str(v))
        return arguments

    def get_debug_vector(self, command: str, config: str = None):
        args = self.to_args()
        config = str(Path(config or self.path).absolute()).replace("\\", "/")
        return 'std::vector<std::string> arguments{"%s", "%s", ' % (command, config) + ",\n".join(f'"{x}"' for x in args) + '};\n'
    