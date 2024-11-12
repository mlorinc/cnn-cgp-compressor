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
# test_cgp_adapter.py: Python CGP dummy implementation.

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
    