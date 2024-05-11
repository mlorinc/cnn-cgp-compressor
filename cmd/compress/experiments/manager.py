import argparse
from experiments.composite.experiment import MultiExperiment
from experiments.all_layers.experiment import AllLayersExperiment
from experiments.mobilenet.experiment import MobilenetExperiment
from experiments.layer_bypass.experiment import LayerBypassExperiment
from experiments.le_selector.experiment import LeSelectorExperiment
from experiments.worst_case.experiment import WorstCaseExperiment
from experiments.grid_size.experiment import GridSizeExperiment 
from experiments.reversed_single_filter.experiment import ReversedSingleFilterExperiment
from experiments.single_channel.experiment import SingleChannelExperiment
from experiments.single_filter_zero_outter.experiment import SingleFilterZeroOutterExperiment
from experiments.single_filter.experiment import SingleFilterExperiment

import experiments.all_layers.cli as al
import experiments.mobilenet.cli as mn
import experiments.layer_bypass.cli as lb
import experiments.le_selector.cli as le
import experiments.worst_case.cli as wc
import experiments.grid_size.cli as gs
import experiments.reversed_single_filter.cli as rs
import experiments.single_channel.cli as sc
import experiments.single_filter_zero_outter.cli as sf

import experiments.composite.cli as composite_cli

experiments_classes = {
    AllLayersExperiment.name:               AllLayersExperiment,
    MobilenetExperiment.name:               MobilenetExperiment,
    LayerBypassExperiment.name:             LayerBypassExperiment,
    LeSelectorExperiment.name:              LeSelectorExperiment,
    WorstCaseExperiment.name:               WorstCaseExperiment,
    GridSizeExperiment.name:                GridSizeExperiment,
    ReversedSingleFilterExperiment.name:    ReversedSingleFilterExperiment,
    SingleChannelExperiment.name:           SingleChannelExperiment,
    SingleFilterZeroOutterExperiment.name:  SingleFilterZeroOutterExperiment,
    SingleFilterExperiment.name:            SingleFilterExperiment
}

experiment_cli = {
    AllLayersExperiment.name:               al.get_argument_parser,
    MobilenetExperiment.name:               mn.get_argument_parser,
    LayerBypassExperiment.name:             lb.get_argument_parser,
    LeSelectorExperiment.name:              le.get_argument_parser,
    WorstCaseExperiment.name:               wc.get_argument_parser,
    GridSizeExperiment.name:                gs.get_argument_parser,
    ReversedSingleFilterExperiment.name:    rs.get_argument_parser,
    SingleChannelExperiment.name:           sc.get_argument_parser,
    SingleFilterZeroOutterExperiment.name:  sf.get_argument_parser,
    SingleFilterExperiment.name:            lambda x: x,
}

experiment_factories = dict([(name, clazz.with_cli_arguments) for name, clazz in experiments_classes.items()])

def get_experiment_factory(name: str):
    return experiment_factories.get(name)

def get_experiment_class(name: str):
    return experiments_classes.get(name)

def get_experiment_arguments(name: str, parser):
    composite_cli.get_argument_parser(parser)
    experiment_cli.get(name)(parser)
    return parser

def get_base_argument_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--depth", action="store_true", default=False, help="Include depth metric")
    parser.add_argument("-e", "--allowed-mse-error", default=0.15, help="Allowed error when chromosomes will be logged in statistics")
    return parser

def get_pbs_default_arguments_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--time-limit", required=True, help="Time limit for the PBS job")
    parser.add_argument("--template-pbs-file", required=True, help="Path to the template PBS file")
    parser.add_argument("--experiments-folder", default="experiments_folder", help="Experiments folder")
    parser.add_argument("--results-folder", default="results", help="Results folder")
    parser.add_argument("--cpu", type=int, default=None, help="Number of CPUs")
    parser.add_argument("--mem", default="2gb", help="Memory")
    parser.add_argument("--scratch-capacity", default="1gb", help="Scratch capacity")
    parser.add_argument("--e-fitness", default="SE", type=str, help="Error fitness")
    parser.add_argument("--multiplex", action="store_true", help="Use Multiplex Optimisation")
    return parser

def get_train_pbs_argument_parser(parser: argparse.ArgumentParser):
    get_pbs_default_arguments_parser(parser)
    parser.add_argument("--cgp-folder", default="cgp_cpp_project", help="CGP folder")
    return parser
