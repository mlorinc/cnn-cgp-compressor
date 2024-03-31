import torch
from cgp.cgp_adapter import CGP
from models.base import BaseModel
from experiments.experiment import BaseExperiment, conv2d_core, conv2d_outter

class AllLayersExperiment(BaseExperiment):
    name = "all_layers"
    def __init__(self, experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None) -> None:
        super().__init__(experiment_folder, experiment_name or AllLayersExperiment.name, model, cgp, dtype)
        self.add_filters("conv1", 
            [
                conv2d_core([slice(None), slice(None)], 5, 3)
            ], 
            [
                *conv2d_outter([slice(None), slice(None)], 5, 3)
            ])
        self.next_input_combination()
        self.add_filters("conv2", 
            [
                conv2d_core([slice(None), slice(None)], 5, 3)
            ], 
            [
                *conv2d_outter([slice(None), slice(None)], 5, 3)
            ])

def init(experiment_folder: str, model: BaseModel, cgp: CGP, dtype=torch.int8, experiment_name=None):
    return AllLayersExperiment(experiment_folder, model, cgp, dtype, experiment_name=experiment_name)
