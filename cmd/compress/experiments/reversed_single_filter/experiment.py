import torch
from cgp.cgp_adapter import CGP
from models.base import BaseModel
from experiments.experiment import BaseExperiment, conv2d_core, conv2d_outter

class ReversedSingleFilterExperiment(BaseExperiment):
    name = "reversed_single_filter"
    def __init__(self, experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or ReversedSingleFilterExperiment.name, model, cgp, dtype)
        self.add_filters("conv1", 
            [
                *conv2d_outter([0, 0], 5, 3)
            ], 
            [
                conv2d_core([0, 0], 5, 3)
            ])

def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return ReversedSingleFilterExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
