import torch
import subprocess

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
    def __init__(self, binary: str, kernel_count: int, kernel_dimension: int, core_dimension: int, dtype=torch.int8) -> None:
        self._binary = binary
        self._core_dimension = core_dimension
        self._kernel_dimension = kernel_dimension
        self._kernel_size = kernel_dimension ** 2
        self._core_size = core_dimension ** 2
        self._outter_size = self._kernel_size - self._core_size
        self._kernel_count = kernel_count

        self._input_size = kernel_count * self._core_size
        self._output_size = kernel_count * self._kernel_size - self._input_size
        self._inputs = torch.empty(size=(self._input_size, ), dtype=dtype)
        self._expected_values = torch.empty(size=(self._output_size, ), dtype=dtype)
        self._input_position = (0, self._core_size)
        self._output_position = (0, self._outter_size)
        self._weights = None
        self._trained = False
        self._dtype = dtype

    def add_kernel(self, kernel: torch.Tensor):
        if kernel.dtype == torch.qint8:
            kernel = kernel.int()

        core, out = convert_kernel_to_train_dataset(kernel, self._core_dimension)
        self._inputs[self._input_position[0]:self._input_position[1]] = core
        self._expected_values[self._output_position[0]:self._output_position[1]] = out
        self._input_position = (self._input_position[0] + self._core_size, self._input_position[1] + self._core_size)
        self._output_position = (self._output_position[0] + self._outter_size, self._output_position[1] + self._outter_size)

    def train(self):
        process = subprocess.Popen([self._binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        # Send initializing parameters: number of layers to approximate, input size and output size
        process.stdin.write(f"{self._kernel_count} {self._input_size} {self._output_size} ")
        # Send train data to the stdin of the subprocess
        process.stdin.write(" ".join(self._inputs.numpy().astype(str)) + " ")
        # Send target data to the stdin of the subprocess
        process.stdin.write(" ".join(self._expected_values.numpy().astype(str)))
        # Close stdin to signal the end of input
        process.stdin.close()

        capture_weights = False
        for line in iter(process.stdout.readline, ""):
            print("CGP:", line, end="")

            if capture_weights:
                capture_weights = False
                self._weights = torch.Tensor([float(segment) for segment in line.split(", ") if segment.strip() != ""])

            if line.strip() == "weights:":
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
