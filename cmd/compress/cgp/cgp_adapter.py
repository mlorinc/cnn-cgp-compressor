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
    def __init__(self, binary: str, cgp_config: str, dtype=torch.int8) -> None:
        self._binary = binary
        self.config = CGPConfiguration(cgp_config)
        self._input_size = self.config.get_input_count()
        self._output_size = self.config.get_output_count()
        self._inputs = [torch.empty(size=(self._input_size, ), dtype=dtype) * self.config.get_dataset_size()]
        self._expected_values = [torch.empty(size=(self._output_size, ), dtype=dtype) * self.config.get_dataset_size()]
        self._input_position = 0
        self._output_position = 0
        self._item_index = 0
        self._dtype = dtype

    def add_inputs(self, x: torch.Tensor):
        x = x.flatten()
        self._inputs[self._item_index][self._input_position:self._input_position+x.shape[0]] = x
        self._input_position += x.shape[0]

    def add_outputs(self, x: torch.Tensor):
        x = x.flatten()
        self._expected_values[self._item_index][self._output_position:self._output_position+x.shape[0]] = x
        self._output_position += x.shape[0]

    def next_train_item(self):
        self._item_index += 1
        self._input_position = 0
        self._output_position = 0

    def _prepare_cgp_algorithm(self, stream: TextIO):
        for inputs, outputs in zip(self._inputs, self._expected_values):
            # Send train data to the stdin of the subprocess
            stream.write(" ".join(inputs.numpy().astype(str)) + "\n")
            # Send target data to the stdin of the subprocess
            stream.write(" ".join(outputs.numpy().astype(str)) + "\n")

    def create_train_file(self, file: str):
        with open(file, "w") as f:
            self._prepare_cgp_algorithm(f)

    def train(self, new_configration: CGPConfiguration):
        args = [] if new_configration is None else new_configration.to_args()
        process = subprocess.Popen([
            self._binary,
            "train",
            str(self.config.get_dataset_size()),
            self.config._config_file,
           *args
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        input_file = new_configration.get_input_file() if new_configration.has_input_file() else self.config.get_input_file()
        if input_file == "-":
            self._prepare_cgp_algorithm(process.stdin)
            process.stdin.close()
        else:
            self.create_train_file(input_file)

        for stdout_line in iter(process.stdout.readline, ""):
            print("CGP:", stdout_line, end="")

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
            str(self.config.get_dataset_size()),
            config_file or self.config._config_file,
            *solution_arg,
            *args
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        for stdout_line in iter(process.stdout.readline, ""):
            print("CGP:", stdout_line, end="")

        # Wait for the subprocess to finish
        process.wait()
        print("Return code:", process.returncode)
        if process.returncode != 0:
            raise CGPProcessError(process.returncode)
