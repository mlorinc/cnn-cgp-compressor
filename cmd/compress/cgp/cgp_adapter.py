import torch
import subprocess
from typing import TextIO
from cgp.cgp_configuration import CGPConfiguration

class CGPProcessError(Exception):
    def __init__(self, code: int, what: str = None) -> None:
        self.code = code
        self.what = what
        if code == 0:
            raise ValueError("Program which exited with the code 0 is considered to be sucesfull thus this exception is not valid")

    def __str__(self) -> str:
        return f"The CGP process exited with exit code {self.code}." + (f"Reason: {self.what}" if self.what else "")

class CGP(object):
    def __init__(self, binary: str, cgp_config: str = None, dtype=torch.int8) -> None:
        self._binary = binary
        self.config = CGPConfiguration(cgp_config)
        self._input_size = self.config.get_input_count()
        self._output_size = self.config.get_output_count()
        self._dtype = dtype
        self.reset()

    def reset(self):
        self._inputs = [torch.zeros(size=(self._input_size, ), dtype=self._dtype).detach() for _ in range(self.config.get_dataset_size())]
        self._expected_values = [torch.zeros(size=(self._output_size, ), dtype=self._dtype).detach() for _ in range(self.config.get_dataset_size())]
        self._input_wildcards = [0] * self.config.get_dataset_size()
        self._output_wildcards = [0] * self.config.get_dataset_size()
        self._input_position = 0
        self._output_position = 0
        self._item_index = 0        

    def add_inputs(self, x: torch.Tensor):
        x = x.flatten()
        self._inputs[self._item_index][self._input_position:self._input_position+x.shape[0]] = x
        self._input_position += x.shape[0]

    def add_outputs(self, x: torch.Tensor):
        x = x.flatten()
        self._expected_values[self._item_index][self._output_position:self._output_position+x.shape[0]] = x
        self._output_position += x.shape[0]

    def next_train_item(self):
        self._input_wildcards[self._item_index] = self._input_position
        self._output_wildcards[self._item_index] = self._output_position
        self._item_index += 1
        self._input_position = 0
        self._output_position = 0

    def _prepare_cgp_algorithm(self, stream: TextIO):
        for inputs, outputs, input_wildcard, output_wildcard in zip(self._inputs, self._expected_values, self._input_wildcards, self._output_wildcards):
            no_care_input_values = self.config.get_input_count() - input_wildcard
            no_care_output_values = self.config.get_output_count() - output_wildcard
            
            inputs = inputs.numpy()
            outputs = outputs.numpy()

            stream.write(" ".join(inputs[:input_wildcard].astype(str)))
            if no_care_input_values != 0:
                stream.write(" " + " ".join(["x"] * no_care_input_values))
            stream.write("\n")

            stream.write(" ".join(outputs[:output_wildcard].astype(str)))
            if no_care_output_values != 0:
                stream.write(" " + " ".join(["x"] * no_care_output_values))
            stream.write("\n")

    def create_train_file(self, file: str):
        with open(file, "w") as f:
            self._prepare_cgp_algorithm(f)

    def train(self, new_configration: CGPConfiguration):
        new_configration.remove_redundant_attributes(self.config)
        args = [] if new_configration is None else new_configration.to_args()

        input_file = new_configration.get_input_file() if new_configration.has_input_file() else self.config.get_input_file()
        if input_file != "-":
            self.create_train_file(input_file)

        resumed_run = (new_configration.has_start_run() and new_configration.get_start_run() != 0)
        resumed_generation = (new_configration.has_start_generation() and new_configration.get_start_generation() != 0)
        file_mode = "a" if resumed_run or resumed_generation else "w"
        with open(new_configration.get_stdout_file(), file_mode) as stdout, open(new_configration.get_stderr_file(), file_mode) as stderr:
            process = subprocess.Popen([
                self._binary,
                "train",
                self.config._config_file,
            *args
            ], stdin=subprocess.PIPE if input_file == "-" else None, stdout=stdout, stderr=stderr, text=True)
            
            if input_file == "-":
                self._prepare_cgp_algorithm(process.stdin)
                process.stdin.close()

            # for stdout_line in iter(process.stderr.readline, ""):
            #     print("CGP:", stdout_line, end="")

            # Wait for the subprocess to finish
            process.wait()
            print("Return code:", process.returncode)
            if process.returncode != 0:
                raise CGPProcessError(process.returncode)

    def evaluate(self, new_configration: CGPConfiguration = None, solution: str = None, config_file: str = None):
        args = [] if new_configration is None else new_configration.to_args()
        solution_arg = [solution] if solution is not None else []
        process = subprocess.Popen([
            self._binary,
            "evaluate" if solution is None else "evaluate:inline",
            config_file or self.config._config_file,
            *solution_arg,
            *args
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for stdout_line in iter(process.stderr.readline, ""):
            print("CGP:", stdout_line, end="")

        # Wait for the subprocess to finish
        process.wait()
        print("Return code:", process.returncode)
        if process.returncode != 0:
            raise CGPProcessError(process.returncode)
