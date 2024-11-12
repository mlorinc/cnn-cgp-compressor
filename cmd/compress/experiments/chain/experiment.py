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
# experiment.py: Experiment implementation that infers outter weights using core 3x3 weights on every kernel in chain. Unused in the end.

import torch
from cgp.cgp_adapter import CGP
from models.base_model import BaseModel
from experiments.experiment import BaseExperiment, conv2d_core, conv2d_outter

class ChainExperiment(BaseExperiment):
    name = "chain"
    def __init__(self, experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or ChainExperiment.name, model, cgp, dtype)
        self.add_filters("conv1", 
            [
                conv2d_core([0, slice(None)], 5, 3)
            ], 
            [
                *conv2d_outter([0, slice(None)], 5, 3)
            ])
        self.add_filters("conv2", 
            [
                conv2d_core([slice(None), 0], 5, 3)
            ], 
            [
                *conv2d_outter([slice(None), 0], 5, 3)
            ])

def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return ChainExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
