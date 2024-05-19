import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.composite.experiment import MultiExperiment
from experiments.experiment import Experiment
from models.selector import FilterSelector, ByteSelector, ValuesSelector, FilterSelectorCombination, FilterSelectorCombinations
from typing import List
from parse import parse

class WorstCaseExperiment(MultiExperiment):
    name = "worst_case"
    thresholds = [0]
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP,
                 dtype=torch.int8, 
                 mse_thresholds=thresholds,
                 prefix="",
                 suffix="",
                 prepare=True,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        
        if prepare:
            self._prepare_filters()

    def _prepare_filters(self):
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}mse_{mse}_{self.args['rows']}_{self.args['cols']}{self.suffix}", self._get_filters()):
                experiment.config.set_mse_threshold(mse)
                experiment.config.set_row_count(self.args["rows"])
                experiment.config.set_col_count(self.args["cols"])
                experiment.config.set_look_back_parameter(self.args["cols"])

    def create_experiment_from_name(self, config: CGPConfiguration):
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error, **self.args)
        result = parse("{prefix}mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name:", name)
        
        mse = int(result["mse"])
        rows = int(result["rows"])
        cols = int(result["cols"])
        experiment.config.set_mse_threshold(mse)
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        return experiment

    def _get_filters(self):
        combinations = FilterSelectorCombinations()
        combination = FilterSelectorCombination()
        # list(range(-128, 128, 1))
        # [-128, 0] + [2**i for i in range(1, 7, 1)])]
        combination.add(FilterSelector("conv1", [ValuesSelector(list(range(0, 65)))], [ValuesSelector(list(range(-128, 128, 1)))]*1))
        combinations.add(combination)
        return combinations

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return WorstCaseExperiment(config, model_adapter, cgp, args,
                                   layer_names=args.layer_names,
                                   prefix=args.prefix,
                                   suffix=args.suffix,
                                   mse_thresholds=args.mse_thresholds,
                                   batches=args.batches)
