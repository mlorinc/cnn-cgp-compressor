import torch
from cgp.cgp_adapter import CGP

class TestCGP(CGP):
    def __init__(self, *args, ruin_tensors=False) -> None:
        self._kernels = []
        self._ruin_tensors = ruin_tensors
    def add_kernel(self, kernel: torch.Tensor):
        return self._kernels.append(kernel if not self._ruin_tensors else torch.randn(kernel.shape))
    def train(self):
        pass
    def get_kernels(self):
        return self._kernels
    