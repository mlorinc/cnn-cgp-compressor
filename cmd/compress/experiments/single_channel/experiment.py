import torch
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from models.adapters.model_adapter import ModelAdapter
from experiments.multi_experiment import MultiExperiment
from models.quantization import conv2d_selector
import argparse

class SingleChannelExperiment(MultiExperiment):
    layer_name = "conv1"
    name = "single_channel"
    thresholds = [0, 1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 200]
    rows_per_filter = 5
    channel = 0
    def __init__(self, 
                 config: CGPConfiguration, 
                 model_adapter: ModelAdapter, 
                 cgp: CGP, 
                 args,
                 dtype=torch.int8, 
                 layer_name=layer_name, 
                 channel=channel, 
                 prefix="", 
                 suffix="", 
                 mse_thresholds=thresholds,
                 rows_per_filter=rows_per_filter,
                 **kwargs) -> None:
        super().__init__(config, model_adapter, cgp, args, dtype, **kwargs)
        self.mse_thresholds = mse_thresholds
        self.prefix = prefix
        self.suffix = suffix
        self.rows_per_filter = rows_per_filter
        self.layer_name = layer_name
        self.channel = channel
        self._prepare_filters()

    def _prepare_filters(self):
        layer = self._model_adapter.get_layer(self.layer_name)
        single_cell_size = self.rows_per_filter * layer.out_channels
        for mse in self.mse_thresholds:
            for experiment in self.create_experiment(f"{self.prefix}{self.layer_name}_mse_{mse}{self.suffix}", self._get_filter(self.layer_name, self.channel)):
                experiment.config.set_mse_threshold(mse)
                experiment.config.set_row_count(single_cell_size)
                experiment.config.set_col_count(15)
                experiment.config.set_look_back_parameter(15)
                experiment.config.set_patience(max(1000000, int(8 / experiment.config.get_population_max() * experiment.config.get_patience())))
                experiment.config.set_mse_chromosome_logging_threshold(max(SingleChannelExperiment.thresholds))

    def _get_filter(self, layer_name: str, channel_i: int):
        return conv2d_selector(layer_name, [slice(None), channel_i], 5, 3)

    @staticmethod
    def get_argument_parser(parser: argparse._SubParsersAction):
        parser.add_argument("--layer-name", default=SingleChannelExperiment.layer_name, help="Name of the layer")
        parser.add_argument("--channel", type=int, default=SingleChannelExperiment.channel, help="Channel index")
        parser.add_argument("--prefix", default="", help="Prefix for experiment names")
        parser.add_argument("--suffix", default="", help="Suffix for experiment names")
        parser.add_argument("--mse-thresholds", nargs="+", type=int, default=SingleChannelExperiment.thresholds, help="List of MSE thresholds")
        parser.add_argument("--rows_per_filter", type=int, default=SingleChannelExperiment.rows_per_filter, help="CGP rows used per filter for circuit design")
        MultiExperiment.get_argument_parser(parser)
        return parser

    @staticmethod
    def new(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args):
        return SingleChannelExperiment(config, model_adapter, cgp, args,
                                    layer_name=args.layer_name,
                                    channel=args.channel,
                                    prefix=args.prefix,
                                    suffix=args.suffix,
                                    mse_thresholds=args.mse_thresholds,
                                    rows_per_filter=args.rows_per_filter,
                                    batches=args.batches)
