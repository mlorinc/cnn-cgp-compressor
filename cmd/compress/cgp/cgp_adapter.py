import torch
import subprocess
from typing import TextIO
from cgp.cgp_configuration import CGPConfiguration
from pathlib import Path
import os

class CGPProcessError(Exception):
    """
    Exception raised for errors in the CGP process.

    Attributes:
        code (int): Exit code of the CGP process.
        what (str): Optional description of the error.
    """    
    def __init__(self, code: int, what: str = None) -> None:
        """
        Initializes CGPProcessError with an exit code and an optional description.

        Args:
            code (int): Exit code of the CGP process.
            what (str): Optional description of the error.
        
        Raises:
            ValueError: If the exit code is 0, indicating a successful program execution.
        """        
        self.code = code
        self.what = what
        if code == 0:
            raise ValueError("Program which exited with the code 0 is considered to be sucesfull thus this exception is not valid")

    def __str__(self) -> str:
        return f"The CGP process exited with exit code {self.code}." + (f"Reason: {self.what}" if self.what else "")

class CGP(object):
    """
    Adapter class providing an API to the CGP C++ module.
    """
    def __init__(self, binary: str, dtype=torch.int8) -> None:
        """
        Initializes the CGP adapter.

        Args:
            binary (str): Path to the CGP C++ binary.
            dtype (torch.dtype): Data type for the tensors.
        """        
        if binary is None:
            self._binary = None
        else:
            self._binary = Path(binary)
        self.config = None
        self._dtype = dtype

    def setup(self, config: CGPConfiguration):
        """
        Sets up the CGP configuration and initializes tensors for inputs and expected values.

        Args:
            config (CGPConfiguration): CGP configuration instance.
        """        
        self.config = config
        input_size = int(self.config.get_input_count())
        output_size = int(self.config.get_output_count())
        print(f"CGP INIT: input_size={input_size}, output_size={output_size}, dtype={self._dtype}, dataset_size={config.get_dataset_size()}")
        self._inputs = [torch.zeros(size=(input_size, ), dtype=self._dtype).detach() for _ in range(config.get_dataset_size())]
        self._expected_values = [torch.zeros(size=(output_size, ), dtype=self._dtype).detach() for _ in range(config.get_dataset_size())]
        self._input_wildcards = [0] * config.get_dataset_size()
        self._output_wildcards = [0] * config.get_dataset_size()
        self._input_position = 0
        self._output_position = 0
        self._item_index = 0        

    def add_inputs(self, x: torch.Tensor):
        """
        Adds input tensors to the current selected dataset.

        Args:
            x (torch.Tensor): Input tensor.
        """        
        x = x.flatten()
        size = min(self._input_position+x.shape[0], self.config.get_input_count()) - self._input_position
        self._inputs[self._item_index][self._input_position:self._input_position+size] = x[:size]
        self._input_position += size

    def add_outputs(self, x: torch.Tensor):
        """
        Adds output tensors to the current dataset.

        Args:
            x (torch.Tensor): Output tensor.
        """        
        x = x.flatten()
        self._expected_values[self._item_index][self._output_position:self._output_position+x.shape[0]] = x
        self._output_position += x.shape[0]

    def next_train_item(self):
        """
        Moves to the next training dataset, saving the positions of the current item.
        """        
        self._input_wildcards[self._item_index] = self._input_position
        self._output_wildcards[self._item_index] = self._output_position
        self._item_index += 1
        self._input_position = 0
        self._output_position = 0

    def _dump_train_weights(self, stream: TextIO):
        """
        Dumps training weights to a given stream.

        Args:
            stream (TextIO): Output stream to write the weights.
        """        
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
        """
        Creates a training file with the current training data.

        Args:
            file (str): Path to the training file.
        
        Raises:
            ValueError: If the file path is stdin ("-").
        """        
        if file == "-":
            raise ValueError("cannot write train data to stdin")
        with open(file, "w") as f:
            self._dump_train_weights(f)

    def get_cli_arguments(self, command: str = "train", other_args=[], cwd: str = None):
        """
        Constructs the command line arguments for the CGP process.

        Args:
            command (str): Command to execute.
            other_args (list): Additional arguments.
            cwd (str): Current working directory.

        Returns:
            list: List of command line arguments.
        """        
        args = self.config.to_args()

        return [str(self._binary), command, str(self.config.path), *other_args, *args]

    def _execute(self, command: str = "train", mode="w", other_args=[], cwd: str = None):
        """
        Executes a CGP command.

        Args:
            command (str): Command to execute.
            mode (str): Mode for opening the output streams.
            other_args (list): Additional arguments.
            cwd (str): Current working directory.

        Raises:
            CGPProcessError: If the CGP process exits with a non-zero code.
        """        
        args = self.get_cli_arguments(command=command, other_args=other_args, cwd=cwd)
        print(args)
        with self.config.open_stdout(mode) as stdout, self.config.open_stderr(mode) as stderr:
            process = subprocess.Popen(args, stdout=stdout, stderr=None, text=True, cwd=os.getcwd())
            process.wait()
            print("Return code:", process.returncode)
            if process.returncode != 0:
                raise CGPProcessError(process.returncode)

    def train(self):
        """
        Trains the CGP model.

        Returns:
            None
            
        Raises:
            CGPProcessError: If the CGP process exits with a non-zero code.            
        """        
        file_mode = "a" if self.config.should_resume_evolution() else "w"
        return self._execute(command="train", mode=file_mode)

    def evaluate(self, solution: str = None):
        """
        Evaluates the CGP model.

        Args:
            solution (str): Path to the solution file.

        Returns:
            None
            
        Raises:
            CGPProcessError: If the CGP process exits with a non-zero code.            
        """        
        solution_arg = [solution] if solution is not None else []
        return self._execute(command="evaluate", other_args=solution_arg)

    def evaluate_all(self):
        """
        Evaluates all CGP models.

        Returns:
            None
            
        Raises:
            CGPProcessError: If the CGP process exits with a non-zero code.            
        """        
        return self._execute(command="evaluate:all")
    
    def evaluate_chromosomes(self, gate_statistics_file: str, *args, config_path: str = None, train_weights: str = None, chromosome_file: str = None, output_statistics: str = None, output_weights: str = None, gate_parameters_file: str = None):
        """
        Evaluates specific chromosomes of the CGP model.

        Args:
            gate_statistics_file (str): File to write gate statistics.
            args: Additional arguments.
            config_path (str): Path to the configuration file.
            train_weights (str): Path to the training weights file.
            chromosome_file (str): Path to the chromosome file.
            output_statistics (str): Path to the output statistics file.
            output_weights (str): Path to the output weights file.
            gate_parameters_file (str): Path to the gate parameters file.

        Returns:
            None
            
        Raises:
            CGPProcessError: If the CGP process exits with a non-zero code.            
        """        
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
        """
        Evaluates a specific chromosome of the CGP model.

        Args:
            chromosome (str): Path to the chromosome file.

        Returns:
            None

        Raises:
            CGPProcessError: If the CGP process exits with a non-zero code.
        """        
        return self._execute(command="evaluate:chromosome", other_args=[chromosome])
