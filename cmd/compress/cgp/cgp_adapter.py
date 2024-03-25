import torch
import subprocess
from typing import TextIO

def convert_kernel_to_train_dataset(kernel: torch.Tensor, core_size: int):
    # Ensure the core size is valid
    if core_size % 2 == 0 or core_size > min(kernel.shape):
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")

    skip = (core_size - 1) // 2

    # Extract the core
    core_array = kernel[skip:skip + core_size, skip:skip + core_size]
    outer_array = []
    row = 0
    for _ in range(skip):
        outer_array.append(kernel[row, :])
        row += 1

    for _ in range(core_size):
        outer_array.append(kernel[row, 0:skip])
        outer_array.append(kernel[row, skip+core_size:])
        row += 1
    
    for _ in range(skip):
        outer_array.append(kernel[row, :])
        row += 1

    return core_array.flatten(), torch.concatenate(outer_array)

def create_kernel(core: torch.Tensor, outter: torch.Tensor, kernel_shape: torch.Size):
    core_size = core.shape[0]
    # Ensure the core size is valid
    if core_size % 2 == 0 or core_size > min(kernel_shape):
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")

    skip = (core_size - 1) // 2

    kernel = torch.Tensor(size=kernel_shape)
    kernel[skip:skip + core_size, skip:skip + core_size] = core

    row = 0
    outter_index = 0
    for _ in range(skip):
        kernel[row, :] = outter[outter_index:outter_index+kernel_shape[1]]
        row += 1
        outter_index += kernel_shape[1]

    for _ in range(core_size):
        kernel[row, 0:skip] = outter[outter_index:outter_index + skip]
        outter_index += skip
        kernel[row, skip+core_size:] = outter[outter_index:outter_index + skip]
        outter_index += skip
        row += 1
    
    for _ in range(skip):
        kernel[row, :] = outter[outter_index:outter_index+kernel_shape[1]]
        row += 1
        outter_index += kernel_shape[1]

    return kernel

class CGP(object):
    def __init__(self, binary: str, train_dataset_size: int, input_size: int, output_size: int, dtype=torch.int8) -> None:
        self._binary = binary
        self.train_dataset_size = train_dataset_size

        self._input_size = input_size
        self._output_size = output_size
        self._inputs = [torch.empty(size=(self._input_size, ), dtype=dtype) * train_dataset_size]
        self._expected_values = [torch.empty(size=(self._output_size, ), dtype=dtype) * train_dataset_size]
        self._input_position = 0
        self._output_position = 0
        self._item_index = 0
        self._weights = None
        self._trained = False
        self._dtype = dtype

    def add_kernel(self, kernel: torch.Tensor, core_dimension: int):
        if kernel.dtype == torch.qint8:
            kernel = kernel.int()

        core, out = convert_kernel_to_train_dataset(kernel, core_dimension)
        core_dimension = core.shape[0]
        out_dimension = out.shape[0]

        self._inputs[self._item_index][self._input_position:self._input_position+core_dimension] = core
        self._expected_values[self._item_index][self._output_position:self._output_position+out_dimension] = out
        self._input_position += core_dimension
        self._output_position += out_dimension

    def next_train_item(self):
        self._item_index += 1
        self._input_position = 0
        self._output_position = 0

    def _prepare_cgp_algorithm(self, stream: TextIO):
        stream.write(f"{self.train_dataset_size} {self._input_size} {self._output_size}\n")
        for inputs, outputs in zip(self._inputs, self._expected_values):
            # Send train data to the stdin of the subprocess
            stream.write(" ".join(inputs.numpy().astype(str)) + "\n")
            # Send target data to the stdin of the subprocess
            stream.write(" ".join(outputs.numpy().astype(str)) + "\n")

    def create_train_file(self, file: str):
        with open(file, "w") as f:
            self._prepare_cgp_algorithm(f)

    def train(self):
        process = subprocess.Popen([self._binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._prepare_cgp_algorithm(process.stdin)
        # Close stdin to signal the end of input
        process.stdin.close()

        capture_weights = False
        for stdout_line in iter(process.stdout.readline, ""):
            print("CGP:", stdout_line, end="")

            if capture_weights:
                capture_weights = False
                self._weights = torch.Tensor([float(segment) for segment in stdout_line.split(", ") if segment.strip() != ""])

            if stdout_line.strip() == "weights:":
                capture_weights = True

        # Wait for the subprocess to finish
        process.wait()
        print("Return code:", process.returncode)
        self._trained = True

    def get_kernels(self):
        if not self._trained:
            raise ValueError("the CGP has not been trained")
        kernels = []
        for i in range(self._kernel_count):
            core = self._inputs[i*self._core_size:i*self._core_size+self._core_size]
            outter = self._weights[i*self._outter_size:i*self._outter_size+self._outter_size]
            core = core.reshape((self._core_dimension, self._core_dimension))
            kernel = create_kernel(core, outter, torch.Size((self._kernel_dimension, self._kernel_dimension)))

            if self._dtype == torch.int8:
                kernel = kernel.int()
            kernels.append(kernel)
        return kernels
