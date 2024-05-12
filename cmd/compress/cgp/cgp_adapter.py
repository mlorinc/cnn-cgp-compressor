import torch
import subprocess
from typing import TextIO
from cgp.cgp_configuration import CGPConfiguration
from pathlib import Path
import os

class CGPProcessError(Exception):
    def __init__(self, code: int, what: str = None) -> None:
        self.code = code
        self.what = what
        if code == 0:
            raise ValueError("Program which exited with the code 0 is considered to be sucesfull thus this exception is not valid")

    def __str__(self) -> str:
        return f"The CGP process exited with exit code {self.code}." + (f"Reason: {self.what}" if self.what else "")

class CGP(object):
    def __init__(self, binary: str, dtype=torch.int8) -> None:
        if binary is None:
            self._binary = None
        else:
            self._binary = Path(binary)
        self.config = None
        self._dtype = dtype

    def setup(self, config: CGPConfiguration):
        self.config = config
        input_size = self.config.get_input_count()
        output_size = self.config.get_output_count()
        self._inputs = [torch.zeros(size=(input_size, ), dtype=self._dtype).detach() for _ in range(config.get_dataset_size())]
        self._expected_values = [torch.zeros(size=(output_size, ), dtype=self._dtype).detach() for _ in range(config.get_dataset_size())]
        self._input_wildcards = [0] * config.get_dataset_size()
        self._output_wildcards = [0] * config.get_dataset_size()
        self._input_position = 0
        self._output_position = 0
        self._item_index = 0        

    def add_inputs(self, x: torch.Tensor):
        x = x.flatten()
        size = min(self._input_position+x.shape[0], self.config.get_input_count()) - self._input_position
        self._inputs[self._item_index][self._input_position:self._input_position+size] = x[:size]
        self._input_position += size

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

    def _dump_train_weights(self, stream: TextIO):
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
        if file == "-":
            raise ValueError("cannot write train data to stdin")
        with open(file, "w") as f:
            self._dump_train_weights(f)

    def get_cli_arguments(self, command: str = "train", other_args=[], cwd: str = None):
        args = self.config.to_args()

        return [str(self._binary), command, str(self.config.path), *other_args, *args]

    def _execute(self, command: str = "train", mode="w", other_args=[], cwd: str = None):
        args = self.get_cli_arguments(command=command, other_args=other_args, cwd=cwd)
        print(args)
        with self.config.open_stdout(mode) as stdout, self.config.open_stderr(mode) as stderr:
            process = subprocess.Popen(args, stdout=stdout, stderr=None, text=True, cwd=os.getcwd())
            process.wait()
            print("Return code:", process.returncode)
            if process.returncode != 0:
                raise CGPProcessError(process.returncode)

    def train(self):
        file_mode = "a" if self.config.should_resume_evolution() else "w"
        return self._execute(command="train", mode=file_mode)

    def evaluate(self, solution: str = None):
        solution_arg = [solution] if solution is not None else []
        return self._execute(command="evaluate", other_args=solution_arg)

    def evaluate_all(self):
        return self._execute(command="evaluate:all")
    
    def evaluate_chromosomes(self, gate_statistics_file: str, *args, config_path: str = None, train_weights: str = None, chromosome_file: str = None, output_statistics: str = None, output_weights: str = None, gate_parameters_file: str = None):
        config = self.config.clone() if self.config is not None else CGPConfiguration(config_path)
        config.set_input_file(train_weights or config.get_input_file())
        config.set_cgp_statistics_file(chromosome_file or config.get_cgp_statistics_file())
        config.set_output_file(output_statistics or config.get_output_file())
        config.set_train_weights_file(output_weights or config.get_train_weights_file())
        config.set_gate_parameters_file(gate_parameters_file or config.get_gate_parameters_file())
        
        old_config, self.config = self.config, config
        try:
            return self._execute(command="evaluate:chromosomes", other_args=[str(gate_statistics_file)] + list(*args))
        finally:
            self.config = old_config
    
    def evaluate_chromosome(self, chromosome: str):
        return self._execute(command="evaluate:chromosome", other_args=[chromosome])
