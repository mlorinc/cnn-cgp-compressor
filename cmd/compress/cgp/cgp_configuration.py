from pathlib import Path, PureWindowsPath
from typing import TextIO, Optional, Union
import contextlib
import subprocess
import os
import copy

class CGPConfiguration:
    ignored_arguments = set(["stdout", "stderr"])
    COMMAND_MSE_CHROMOSOME_LOGGING_THRESHOLD = "mse_chromosome_logging_threshold"
    COMMAND_TRAIN_WEIGHTS_FILE = "train_weights_file"
    COMMAND_GATE_PARAMETERS_FILE = "gate_parameters_file"
    COMMAND_STDERR = "stderr"
    COMMAND_STDOUT = "stdout"
    COMMAND_START_GENERATION = "start_generation"
    COMMAND_START_RUN = "start_run"
    COMMAND_ENERGY_EARLY_STOP = "energy_early_stop"
    COMMAND_MSE_EARLY_STOP = "mse_early_stop"
    COMMAND_DELAY_EARLY_STOP = "delay_early_stop"
    COMMAND_DEPTH_EARLY_STOP = "depth_early_stop"
    COMMAND_GATE_COUNT_EARLY_STOP = "gate_count_early_stop"
    COMMAND_PATIENCE = "patience"
    COMMAND_STARTING_SOLUTION = "starting_solution"
    COMMAND_MSE_THRESHOLD = "mse_threshold"
    COMMAND_DATASET_SIZE = "dataset_size"
    COMMAND_COL_COUNT = "col_count"
    COMMAND_ROW_COUNT = "row_count"
    COMMAND_NUMBER_OF_RUNS = "number_of_runs"
    COMMAND_LOOK_BACK_PARAMETER = "look_back_parameter"
    COMMAND_MUTATION_MAX = "mutation_max"
    COMMAND_LEARNING_RATE = "learning_rate"
    COMMAND_LEARNING_RATE_FILE = "learning_rate_file"
    COMMAND_FUNCTION_COUNT = "function_count"
    COMMAND_FUNCTION_INPUT_ARITY = "function_input_arity"
    COMMAND_FUNCTION_OUTPUT_ARITY = "function_output_arity"
    COMMAND_INPUT_COUNT = "input_count"
    COMMAND_OUTPUT_COUNT = "output_count"
    COMMAND_POPULATION_MAX = "population_max"
    COMMAND_GENERATION_COUNT = "generation_count"
    COMMAND_PERIODIC_LOG_FREQUENCY = "periodic_log_frequency"
    COMMAND_INPUT_FILE = "input_file"
    COMMAND_OUTPUT_FILE = "output_file"
    COMMAND_CGP_STATISTICS_FILE = "cgp_statistics_file"

    ARGUMENTS = {
        "periodic-log-frequency": {"help": "The log frequency in the CGP algorithm.", "type": int, "attribute": COMMAND_PERIODIC_LOG_FREQUENCY},
        "function-input-arity": {"help": "The input arity of functions.", "type": int, "attribute": COMMAND_FUNCTION_INPUT_ARITY},
        "function-output-arity": {"help": "The output arity of functions.", "type": int, "attribute": COMMAND_FUNCTION_OUTPUT_ARITY},
        "output-count": {"help": "The number of output pins in the CGP.", "type": int, "attribute": COMMAND_OUTPUT_COUNT},
        "input-count": {"help": "The number of input pins in the CGP.", "type": int, "attribute": COMMAND_INPUT_COUNT},
        "population-max": {"help": "The maximum population size in the CGP algorithm.", "type": int, "attribute": COMMAND_POPULATION_MAX},
        "mutation-max": {"help": "The maximum mutation value in the CGP algorithm.", "type": float, "attribute": COMMAND_MUTATION_MAX},
        "row-count": {"help": "The number of rows in the CGP grid.", "type": int, "attribute": COMMAND_ROW_COUNT},
        "col-count": {"help": "The number of columns in the CGP grid.", "type": int, "attribute": COMMAND_COL_COUNT},
        "look-back-parameter": {"help": "The look-back parameter in the CGP algorithm.", "type": int, "attribute": COMMAND_LOOK_BACK_PARAMETER},
        "generation-count": {"help": "The maximum number of generations in the CGP algorithm.", "type": int, "attribute": COMMAND_GENERATION_COUNT},
        "number-of-runs": {"help": "The number of runs in the CGP algorithm.", "type": int, "attribute": COMMAND_NUMBER_OF_RUNS},
        "function-count": {"help": "The number of functions in the CGP algorithm.", "type": int, "attribute": COMMAND_FUNCTION_COUNT},
        "input-file": {"help": "A path to a file with input data.", "type": str, "attribute": COMMAND_INPUT_FILE},
        "output-file": {"help": "A path to a file to create which contains output of the CGP process.", "type": str, "attribute": COMMAND_OUTPUT_FILE},
        "cgp-statistics-file": {"help": "A path where CGP statistics will be saved.", "type": str, "attribute": COMMAND_CGP_STATISTICS_FILE},
        "gate-parameters-file": {"help": "A path where gate parameters are stored.", "type": str, "attribute": COMMAND_GATE_PARAMETERS_FILE},
        "train-weights-file": {"help": "A path where trained weights parameters will be stored.", "type": str, "attribute": COMMAND_TRAIN_WEIGHTS_FILE},
        "mse-threshold": {"help": "Mean Squared Error threshold after optimization is focused on minimizing energy.", "type": int, "attribute": COMMAND_MSE_THRESHOLD},
        "dataset-size": {"help": "The CGP dataset size.", "type": int, "attribute": COMMAND_DATASET_SIZE},
        "start-generation": {"help": "Used in case CGP evolution is resumed.", "type": int, "attribute": COMMAND_START_GENERATION},
        "start-run": {"help": "Used in case CGP evolution is resumed.", "type": int, "attribute": COMMAND_START_RUN},
        "starting-solution": {"help": "Used in case CGP evolution is resumed.", "type": str, "attribute": COMMAND_STARTING_SOLUTION},
        "patience": {"help": "Value indicating after how many generations CGP will come to stop.", "type": int, "attribute": COMMAND_PATIENCE},
        "mse-early-stop": {"help": "Value indicating stop condition for parameter of approximation error.", "type": float, "attribute": COMMAND_MSE_EARLY_STOP},
        "energy-early-stop": {"help": "Value indicating stop condition for parameter of energy usage.", "type": float, "attribute": COMMAND_ENERGY_EARLY_STOP},
        "delay-early-stop": {"help": "Value indicating stop condition for parameter of delay.", "type": float, "attribute": COMMAND_DELAY_EARLY_STOP},
        "depth-early-stop": {"help": "Value indicating stop condition for parameter of depth.", "type": int, "attribute": COMMAND_DEPTH_EARLY_STOP},
        "gate-count-early-stop": {"help": "Value indicating stop condition for parameter of gate count.", "type": int, "attribute": COMMAND_GATE_COUNT_EARLY_STOP},
        "mse-chromosome-logging-threshold": {"help": "Logging threshold when chromosomes with error less than value will start being printed in CSV logs as serialized strings.", "type": int, "attribute": COMMAND_MSE_CHROMOSOME_LOGGING_THRESHOLD},
        "learning-rate": {"help": "Average learning rate for the CGP model. When average learning rate drops below the value, the whole process is termianted.", "type": float, "attribute": COMMAND_LEARNING_RATE},
        "learning-rate-file": {"help": "Path to a file that will hold learning rate statistics.", "type": str, "attribute": COMMAND_LEARNING_RATE_FILE},
    }

    @staticmethod
    def get_cgp_arguments(parser):
        for command, metadata in CGPConfiguration.ARGUMENTS.items():
            parser.add_argument(f"--{command}", type=metadata["type"], help=metadata["help"], required=False)        

    def __init__(self, config_file: Optional[Union[Path, str]] = None):
        self._extra_attributes = {}
        self._attributes = {}
        self.path = None
        if config_file:
            self.path = Path(config_file)
            self.load(config_file)

    def parse_arguments(self, args):
        for argument_name, metadata in self.ARGUMENTS.items():
            if metadata["attribute"] in args:
                value = args[metadata["attribute"]]
                if value is not None:
                    self.set_attribute(metadata["attribute"], value)

    def clone(self, new_config_file: str = None):
        cloned_instance = CGPConfiguration()
        cloned_instance.path = new_config_file or self.path
        cloned_instance._attributes = copy.deepcopy(self._attributes)
        cloned_instance._extra_attributes = copy.deepcopy(self._extra_attributes)
        return cloned_instance

    def load(self, config_file: str = None):
        if config_file is None and self.path is None:
            raise ValueError(
                "either config file must be passed to the load function or the class constructor must have been provided a configuration file as argument"
            )
        with open(config_file or self.path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines
                if line != "":
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
                if value is None:
                    continue
                if key.endswith("_file"):
                    value = self._path_to_string(value)                
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
        new_extra_dict = dict()
        if self.COMMAND_START_RUN in self._extra_attributes:
            if self._extra_attributes[self.COMMAND_START_RUN] is not None:
                new_extra_dict[self.COMMAND_START_RUN] = self._extra_attributes[self.COMMAND_START_RUN]
            del self._extra_attributes[self.COMMAND_START_RUN]

        if self.COMMAND_START_GENERATION in self._extra_attributes:
            if self._extra_attributes[self.COMMAND_START_GENERATION] is not None:
                new_extra_dict[self.COMMAND_START_GENERATION] = self._extra_attributes[self.COMMAND_START_GENERATION]
            del self._extra_attributes[self.COMMAND_START_GENERATION]

        self._attributes = {**self._attributes, **self._extra_attributes}
        self._extra_attributes = new_extra_dict

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
        return self._extra_attributes.get(name) or self._attributes.get(name, None)

    def get_learning_rate_file(self):
        return self.get_attribute(self.COMMAND_LEARNING_RATE_FILE)

    def get_learning_rate(self):
        return self.get_attribute(self.COMMAND_LEARNING_RATE)

    def get_mse_chromosome_logging_threshold(self):
        return self.get_attribute(self.COMMAND_MSE_CHROMOSOME_LOGGING_THRESHOLD)

    def get_train_weights_file(self):
        return self.get_attribute(self.COMMAND_TRAIN_WEIGHTS_FILE)

    def get_gate_parameters_file(self):
        return self.get_attribute(self.COMMAND_GATE_PARAMETERS_FILE)

    def get_stderr_file(self):
        return self.get_attribute(self.COMMAND_STDERR)

    def get_stdout_file(self):
        return self.get_attribute(self.COMMAND_STDOUT)

    def get_start_generation(self):
        return self.get_attribute(self.COMMAND_START_GENERATION)

    def get_start_run(self):
        return self.get_attribute(self.COMMAND_START_RUN)

    def get_energy_early_stop(self):
        return self.get_attribute(self.COMMAND_ENERGY_EARLY_STOP)

    def get_mse_early_stop(self):
        return self.get_attribute(self.COMMAND_MSE_EARLY_STOP)

    def get_delay_early_stop(self):
        return self.get_attribute(self.COMMAND_DELAY_EARLY_STOP)

    def get_depth_early_stop(self):
        return self.get_attribute(self.COMMAND_DEPTH_EARLY_STOP)

    def get_gate_count_early_stop(self):
        return self.get_attribute(self.COMMAND_GATE_COUNT_EARLY_STOP)

    def get_patience(self):
        return self.get_attribute(self.COMMAND_PATIENCE)

    def get_starting_solution(self):
        return self.get_attribute(self.COMMAND_STARTING_SOLUTION)

    def get_mse_threshold(self):
        return self.get_attribute(self.COMMAND_MSE_THRESHOLD)

    def get_dataset_size(self):
        return self.get_attribute(self.COMMAND_DATASET_SIZE)

    def get_col_count(self):
        return self.get_attribute(self.COMMAND_COL_COUNT)

    def get_row_count(self):
        return self.get_attribute(self.COMMAND_ROW_COUNT)

    def get_number_of_runs(self):
        return self.get_attribute(self.COMMAND_NUMBER_OF_RUNS)

    def get_look_back_parameter(self):
        return self.get_attribute(self.COMMAND_LOOK_BACK_PARAMETER)

    def get_mutation_max(self):
        return self.get_attribute(self.COMMAND_MUTATION_MAX)

    def get_function_count(self):
        return self.get_attribute(self.COMMAND_FUNCTION_COUNT)

    def get_function_input_arity(self):
        return self.get_attribute(self.COMMAND_FUNCTION_INPUT_ARITY)

    def get_function_output_arity(self):
        return self.get_attribute(self.COMMAND_FUNCTION_OUTPUT_ARITY)

    def get_input_count(self):
        return self.get_attribute(self.COMMAND_INPUT_COUNT)

    def get_output_count(self):
        return self.get_attribute(self.COMMAND_OUTPUT_COUNT)

    def get_population_max(self):
        return self.get_attribute(self.COMMAND_POPULATION_MAX)

    def get_generation_count(self):
        return self.get_attribute(self.COMMAND_GENERATION_COUNT)

    def get_periodic_log_frequency(self):
        return self.get_attribute(self.COMMAND_PERIODIC_LOG_FREQUENCY)

    def get_input_file(self):
        return self.get_attribute(self.COMMAND_INPUT_FILE)

    def get_output_file(self):
        return self.get_attribute(self.COMMAND_OUTPUT_FILE)

    def get_cgp_statistics_file(self):
        return self.get_attribute(self.COMMAND_CGP_STATISTICS_FILE)

    def set_attribute(self, attribute, value):
        self._extra_attributes[attribute] = value if not attribute.endswith("_file") else PureWindowsPath(value) if isinstance(value, Path) or isinstance(value, str) else value

    def set_learning_rate_file(self, value):
        self.set_attribute(self.COMMAND_LEARNING_RATE_FILE, value)

    def set_learning_rate(self, value):
        self.set_attribute(self.COMMAND_LEARNING_RATE, value)

    def set_mse_chromosome_logging_threshold(self, value):
        self.set_attribute(self.COMMAND_MSE_CHROMOSOME_LOGGING_THRESHOLD, value)

    def set_train_weights_file(self, value):
        self.set_attribute(self.COMMAND_TRAIN_WEIGHTS_FILE, value)

    def set_gate_parameters_file(self, value):
        self.set_attribute(self.COMMAND_GATE_PARAMETERS_FILE, value)

    def set_stderr_file(self, value):
        self.set_attribute(self.COMMAND_STDERR, value)

    def set_stdout_file(self, value):
        self.set_attribute(self.COMMAND_STDOUT, value)

    def set_start_generation(self, value):
        self.set_attribute(self.COMMAND_START_GENERATION, value)

    def set_start_run(self, value):
        self.set_attribute(self.COMMAND_START_RUN, value)

    def set_energy_early_stop(self, value):
        self.set_attribute(self.COMMAND_ENERGY_EARLY_STOP, value)

    def set_mse_early_stop(self, value):
        self.set_attribute(self.COMMAND_MSE_EARLY_STOP, value)

    def set_delay_early_stop(self, value):
        self.set_attribute(self.COMMAND_DELAY_EARLY_STOP, value)

    def set_depth_early_stop(self, value):
        self.set_attribute(self.COMMAND_DEPTH_EARLY_STOP, value)

    def set_gate_count_early_stop(self, value):
        self.set_attribute(self.COMMAND_GATE_COUNT_EARLY_STOP, value)

    def set_patience(self, value):
        self.set_attribute(self.COMMAND_PATIENCE, value)

    def set_starting_solution(self, value):
        self.set_attribute(self.COMMAND_STARTING_SOLUTION, value)

    def set_mse_threshold(self, value):
        self.set_attribute(self.COMMAND_MSE_THRESHOLD, value)

    def set_dataset_size(self, value):
        self.set_attribute(self.COMMAND_DATASET_SIZE, value)

    def set_col_count(self, value):
        self.set_attribute(self.COMMAND_COL_COUNT, value)

    def set_row_count(self, value):
        self.set_attribute(self.COMMAND_ROW_COUNT, value)

    def set_number_of_runs(self, value):
        self.set_attribute(self.COMMAND_NUMBER_OF_RUNS, value)

    def set_look_back_parameter(self, value):
        self.set_attribute(self.COMMAND_LOOK_BACK_PARAMETER, value)

    def set_mutation_max(self, value):
        self.set_attribute(self.COMMAND_MUTATION_MAX, value)

    def set_function_count(self, value):
        self.set_attribute(self.COMMAND_FUNCTION_COUNT, value)

    def set_function_input_arity(self, value):
        self.set_attribute(self.COMMAND_FUNCTION_INPUT_ARITY, value)

    def set_function_output_arity(self, value):
        self.set_attribute(self.COMMAND_FUNCTION_OUTPUT_ARITY, value)

    def set_input_count(self, value):
        self.set_attribute(self.COMMAND_INPUT_COUNT, value)

    def set_output_count(self, value):
        self.set_attribute(self.COMMAND_OUTPUT_COUNT, value)

    def set_population_max(self, value):
        self.set_attribute(self.COMMAND_POPULATION_MAX, value)

    def set_generation_count(self, value):
        self.set_attribute(self.COMMAND_GENERATION_COUNT, value)

    def set_periodic_log_frequency(self, value):
        self.set_attribute(self.COMMAND_PERIODIC_LOG_FREQUENCY, value)

    def set_input_file(self, value):
        self.set_attribute(self.COMMAND_INPUT_FILE, value)

    def set_output_file(self, value):
        self.set_attribute(self.COMMAND_OUTPUT_FILE, value)

    def set_cgp_statistics_file(self, value):
        self.set_attribute(self.COMMAND_CGP_STATISTICS_FILE, value)

    def delete_attribute(self, name):
        if name in self._attributes:
            del self._attributes[name]
        if name in self._extra_attributes:
            del self._extra_attributes[name]

    def delete_learning_rate_file(self):
        self.delete_attribute(self.COMMAND_LEARNING_RATE_FILE)

    def delete_learning_rate(self):
        self.delete_attribute(self.COMMAND_LEARNING_RATE)

    def delete_mse_chromosome_logging_threshold(self):
        self.delete_attribute(self.COMMAND_MSE_CHROMOSOME_LOGGING_THRESHOLD)

    def delete_train_weights_file(self):
        self.delete_attribute(self.COMMAND_TRAIN_WEIGHTS_FILE)

    def delete_gate_parameters_file(self):
        self.delete_attribute(self.COMMAND_GATE_PARAMETERS_FILE)

    def delete_stderr_file(self):
        self.delete_attribute(self.COMMAND_STDERR)

    def delete_stdout_file(self):
        self.delete_attribute(self.COMMAND_STDOUT)

    def delete_start_generation(self):
        self.delete_attribute(self.COMMAND_START_GENERATION)

    def delete_start_run(self):
        self.delete_attribute(self.COMMAND_START_RUN)

    def delete_energy_early_stop(self):
        self.delete_attribute(self.COMMAND_ENERGY_EARLY_STOP)

    def delete_mse_early_stop(self):
        self.delete_attribute(self.COMMAND_MSE_EARLY_STOP)

    def delete_delay_early_stop(self):
        self.delete_attribute(self.COMMAND_DELAY_EARLY_STOP)

    def delete_depth_early_stop(self):
        self.delete_attribute(self.COMMAND_DEPTH_EARLY_STOP)

    def delete_gate_count_early_stop(self):
        self.delete_attribute(self.COMMAND_GATE_COUNT_EARLY_STOP)

    def delete_patience(self):
        self.delete_attribute(self.COMMAND_PATIENCE)

    def delete_starting_solution(self):
        self.delete_attribute(self.COMMAND_STARTING_SOLUTION)

    def delete_mse_threshold(self):
        self.delete_attribute(self.COMMAND_MSE_THRESHOLD)

    def delete_dataset_size(self):
        self.delete_attribute(self.COMMAND_DATASET_SIZE)

    def delete_col_count(self):
        self.delete_attribute(self.COMMAND_COL_COUNT)

    def delete_row_count(self):
        self.delete_attribute(self.COMMAND_ROW_COUNT)

    def delete_number_of_runs(self):
        self.delete_attribute(self.COMMAND_NUMBER_OF_RUNS)

    def delete_look_back_parameter(self):
        self.delete_attribute(self.COMMAND_LOOK_BACK_PARAMETER)

    def delete_mutation_max(self):
        self.delete_attribute(self.COMMAND_MUTATION_MAX)

    def delete_function_count(self):
        self.delete_attribute(self.COMMAND_FUNCTION_COUNT)

    def delete_function_input_arity(self):
        self.delete_attribute(self.COMMAND_FUNCTION_INPUT_ARITY)

    def delete_function_output_arity(self):
        self.delete_attribute(self.COMMAND_FUNCTION_OUTPUT_ARITY)

    def delete_input_count(self):
        self.delete_attribute(self.COMMAND_INPUT_COUNT)

    def delete_output_count(self):
        self.delete_attribute(self.COMMAND_OUTPUT_COUNT)

    def delete_population_max(self):
        self.delete_attribute(self.COMMAND_POPULATION_MAX)

    def delete_generation_count(self):
        self.delete_attribute(self.COMMAND_GENERATION_COUNT)

    def delete_periodic_log_frequency(self):
        self.delete_attribute(self.COMMAND_PERIODIC_LOG_FREQUENCY)

    def delete_input_file(self):
        self.delete_attribute(self.COMMAND_INPUT_FILE)

    def delete_output_file(self):
        self.delete_attribute(self.COMMAND_OUTPUT_FILE)

    def delete_cgp_statistics_file(self):
        self.delete_attribute(self.COMMAND_CGP_STATISTICS_FILE)

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in self._attributes.items()]
        return "\n".join(attributes)

    def has_attribute(self, name):
        return name in self._attributes or name in self._extra_attributes

    def has_learning_rate_file(self):
        self.has_attribute(self.COMMAND_LEARNING_RATE_FILE)

    def has_learning_rate(self):
        self.has_attribute(self.COMMAND_LEARNING_RATE)

    def has_mse_chromosome_logging_threshold(self):
        return self.has_attribute(self.COMMAND_MSE_CHROMOSOME_LOGGING_THRESHOLD)

    def has_train_weights_file(self):
        return self.has_attribute(self.COMMAND_TRAIN_WEIGHTS_FILE)

    def has_gate_parameters_file(self):
        return self.has_attribute(self.COMMAND_GATE_PARAMETERS_FILE)

    def has_stderr_file(self):
        return self.has_attribute(self.COMMAND_STDERR)

    def has_stdout_file(self):
        return self.has_attribute(self.COMMAND_STDOUT)

    def has_start_generation(self):
        return self.has_attribute(self.COMMAND_START_GENERATION)

    def has_start_run(self):
        return self.has_attribute(self.COMMAND_START_RUN)

    def has_energy_early_stop(self):
        return self.has_attribute(self.COMMAND_ENERGY_EARLY_STOP)

    def has_mse_early_stop(self):
        return self.has_attribute(self.COMMAND_MSE_EARLY_STOP)

    def has_delay_early_stop(self):
        return self.has_attribute(self.COMMAND_DELAY_EARLY_STOP)

    def has_depth_early_stop(self):
        return self.has_attribute(self.COMMAND_DEPTH_EARLY_STOP)

    def has_gate_count_early_stop(self):
        return self.has_attribute(self.COMMAND_GATE_COUNT_EARLY_STOP)

    def has_patience(self):
        return self.has_attribute(self.COMMAND_PATIENCE)

    def has_starting_solution(self):
        return self.has_attribute(self.COMMAND_STARTING_SOLUTION)

    def has_mse_threshold(self):
        return self.has_attribute(self.COMMAND_MSE_THRESHOLD)

    def has_dataset_size(self):
        return self.has_attribute(self.COMMAND_DATASET_SIZE)

    def has_col_count(self):
        return self.has_attribute(self.COMMAND_COL_COUNT)

    def has_row_count(self):
        return self.has_attribute(self.COMMAND_ROW_COUNT)

    def has_number_of_runs(self):
        return self.has_attribute(self.COMMAND_NUMBER_OF_RUNS)

    def has_look_back_parameter(self):
        return self.has_attribute(self.COMMAND_LOOK_BACK_PARAMETER)

    def has_mutation_max(self):
        return self.has_attribute(self.COMMAND_MUTATION_MAX)

    def has_function_count(self):
        return self.has_attribute(self.COMMAND_FUNCTION_COUNT)

    def has_function_input_arity(self):
        return self.has_attribute(self.COMMAND_FUNCTION_INPUT_ARITY)

    def has_function_output_arity(self):
        return self.has_attribute(self.COMMAND_FUNCTION_OUTPUT_ARITY)

    def has_input_count(self):
        return self.has_attribute(self.COMMAND_INPUT_COUNT)

    def has_output_count(self):
        return self.has_attribute(self.COMMAND_OUTPUT_COUNT)

    def has_population_max(self):
        return self.has_attribute(self.COMMAND_POPULATION_MAX)

    def has_generation_count(self):
        return self.has_attribute(self.COMMAND_GENERATION_COUNT)

    def has_periodic_log_frequency(self):
        return self.has_attribute(self.COMMAND_PERIODIC_LOG_FREQUENCY)

    def has_input_file(self):
        return self.has_attribute(self.COMMAND_INPUT_FILE)

    def has_output_file(self):
        return self.has_attribute(self.COMMAND_OUTPUT_FILE)

    def has_cgp_statistics_file(self):
        return self.has_attribute(self.COMMAND_CGP_STATISTICS_FILE)

    def _path_to_string(self, path: Path) -> str:
        return os.path.normpath(os.path.normcase(path)).replace("\\", "/")

    def to_args(self):
        arguments = []
        for k, v in self._extra_attributes.items():
            if k in CGPConfiguration.ignored_arguments:
                continue

            if v is None:
                continue

            if k.endswith("_file"):
                v = self._path_to_string(v)

            if not isinstance(v, str):
                v = str(v)

            key = k.replace("_", "-")
            arguments.append(f"--{key}")
            arguments.append(v)
        return arguments

    def get_debug_vector(self, command: str, config: str = None):
        args = self.to_args()
        config = str(Path(config or self.path).absolute()).replace("\\", "/")
        return 'std::vector<std::string> arguments{"%s", "%s", ' % (command, config) + ",\n".join(f'"{x}"' for x in args) + '};\n'
    