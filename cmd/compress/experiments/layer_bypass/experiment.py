import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.composite.experiment import MultiExperiment
from experiments.experiment import Experiment
from models.quantization import conv2d_selector
from models.selector import FilterSelector, FilterSelectorCombination, FilterSelectorCombinations
import argparse
from typing import List
from parse import parse

class LayerBypassExperiment(MultiExperiment):
    """
    A class to manage layer bypass experiments, extending the MultiExperiment class.
    Layer Bypass, or from thesis refered in general LeNet-5 approximation is an experiment
    examining hypothesis which states it is possible to infer simultaneously with 
    ongoing inference in previous layer.
    """
    name = "layer_bypass"
    thresholds = [250, 150, 100, 50, 25, 15, 10, 0]
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP,
                 dtype=torch.int8, 
                 input_layer_names=["conv1"], 
                 output_layer_names=["conv2"], 
                 mse_thresholds=thresholds,
                 prefix="",
                 suffix="",
                 prepare=True,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, dtype, **kwargs)
        """
        Initialize the LayerBypassExperiment class.

        Args:
            config (CGPConfiguration): Configuration for the CGP.
            model_adapter (ModelAdapter): Adapter for the model.
            cgp (CGP): CGP instance.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to torch.int8.
            input_layer_names (list, optional): List of input layer names. Defaults to ["conv1"].
            output_layer_names (list, optional): List of output layer names. Defaults to ["conv2"].
            mse_thresholds (list, optional): List of MSE thresholds. Defaults to [250, 150, 100, 50, 25, 15, 10, 0].
            prefix (str, optional): Prefix for experiment names. Defaults to "".
            suffix (str, optional): Suffix for experiment names. Defaults to "".
            prepare (bool, optional): Whether to prepare filters automatically. Defaults to True.
            **kwargs: Additional arguments.
        """        
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.input_layer_names = input_layer_names
        self.output_layer_names = output_layer_names
        
        if prepare:
            self._prepare_filters()

    def _prepare_filters(self):
        """
        Prepare filters for the experiments.
        """        
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}mse_{mse}_{self.args['rows']}_{self.args['cols']}{self.suffix}", self._get_filters(self.input_layer_names, self.output_layer_names)):
                experiment.config.set_mse_threshold(int(mse**2 * (16*6*25)))
                # experiment.config.set_mse_chromosome_logging_threshold((30)**2 * (16*6*16+6*16))
                experiment.config.set_row_count(self.args["rows"])
                experiment.config.set_col_count(self.args["cols"])
                experiment.config.set_look_back_parameter(self.args["cols"])

    def create_experiment_from_name(self, config: CGPConfiguration):
        """
        Create an experiment instance from the configuration name.

        Args:
            config (CGPConfiguration): Configuration for the CGP.

        Returns:
            Experiment: The created experiment instance.
        """        
        name = config.path.parent.name
        experiment = Experiment(config, self._model_adapter, self._cgp, self.dtype, depth=self._depth, allowed_mse_error=self._allowed_mse_error, **self.args)
        result = parse("mse_{mse}_{rows}_{cols}", name)
        
        if not result:
            raise ValueError("invalid name " + name)
        
        mse = float(result["mse"])
        rows = int(result["rows"])
        cols = int(result["cols"])
        experiment.config.set_mse_threshold(int(mse**2 * (16*6*25)))
        experiment.config.set_row_count(rows)
        experiment.config.set_col_count(cols)
        experiment.config.set_look_back_parameter(cols)
        experiment.set_feature_maps_combinations(self._get_filters(self.input_layer_names, self.output_layer_names))
        return experiment

    def _get_filters(self, input_layer_names: List[str], output_layer_names: List[str]):
        """
        Get filter selectors for the specified input and output layers.

        Args:
            input_layer_names (List[str]): List of input layer names.
            output_layer_names (List[str]): List of output layer names.

        Returns:
            FilterSelectorCombinations: The filter selector combinations.
        """        
        combinations = FilterSelectorCombinations()
        combination = FilterSelectorCombination()
        for layer_name in input_layer_names:
            combination.add(FilterSelector(layer_name, [(slice(None), slice(None), slice(None), slice(None))], []))
        for layer_name in output_layer_names:
            combination.add(FilterSelector(layer_name, [], [(slice(None), slice(None), slice(None), slice(None))]))
        combinations.add(combination)
        return combinations
