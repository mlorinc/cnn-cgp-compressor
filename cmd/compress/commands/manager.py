import argparse
from commands.debug_model import debug_model
from experiments.experiment import Experiment
from experiments.all_layers.experiment import AllLayersExperiment
from experiments.grid_size.experiment import GridSizeExperiment 
from experiments.reversed_single_filter.experiment import ReversedSingleFilterExperiment
from experiments.single_channel.experiment import SingleChannelExperiment
from experiments.single_filter_zero_outter.experiment import SingleFilterZeroOutterExperiment
from experiments.single_filter.experiment import SingleFilterExperiment
from commands.optimize_prepare_model import optimize_prepare_model
from commands.fix_train_stats import fix_train_statistics
from commands.optimize_model import optimize_model
from commands.evaluate_cgp_model import evaluate_cgp_model
from commands.train_model import train_model
from commands.evaluate_model import evaluate_base_model
from commands.quantize_model import quantize_model
from cgp.cgp_configuration import CGPConfiguration
from typing import List

experiments_classes = {
    AllLayersExperiment.name: AllLayersExperiment,
    GridSizeExperiment.name: GridSizeExperiment,
    ReversedSingleFilterExperiment.name: ReversedSingleFilterExperiment,
    SingleChannelExperiment.name: SingleChannelExperiment,
    SingleFilterZeroOutterExperiment.name: SingleFilterZeroOutterExperiment,
    SingleFilterExperiment.name: SingleFilterExperiment
}

experiment_factories = dict([(name, clazz.new) for name, clazz in experiments_classes.items()])
experiment_commands = ["train", "train-pbs", "evaluate", "fix-train-stats"]

def _register_model_commands(subparsers: argparse._SubParsersAction):
    # model:train
    train_parser = subparsers.add_parser("model:train", help="Train a model")
    train_parser.add_argument("model_name", help="Name of the model to train")
    train_parser.add_argument("model_path", help="Path where trained model will be saved")
    train_parser.add_argument("-b", "--base", type=str, help="Path to the baseline model")

    # model:evaluate
    evaluate_parser = subparsers.add_parser("model:evaluate", help="Evaluate a model")
    evaluate_parser.add_argument("model_name", help="Name of the model to evaluate")
    evaluate_parser.add_argument("model_path", help="Path to the model to evaluate")
    evaluate_parser.add_argument("--weights", nargs="?", type=str, help="", default=[], required=False)

    # model:evaluate
    evaluate_parser = subparsers.add_parser("model:mobilenet_v2", help="Evaluate mobilenet_v2")

    # model:quantize
    quantize_parser = subparsers.add_parser("model:quantize", help="Quantize a model")
    quantize_parser.add_argument("model_name", help="Name of the model to quantize")
    quantize_parser.add_argument("model_path", help="Path where trained model is saved")
    quantize_parser.add_argument("new_path", help="Path of the new quantized model where it will be stored")    

    # model:debug
    debug_parser = subparsers.add_parser("model:debug", help="Debug a model")
    debug_parser.add_argument("model_name", help="Name of the model to quantize")
    debug_parser.add_argument("-m", "--model_path", help="Path where trained model is saved", required=False, default=None) 

def _register_experiment_commands(subparsers: argparse._SubParsersAction, experiment_names: List[str]):
    help_train = "Train a new CGP model to infer mising convolution weights from CNN model. Weights are trained as they are defined by {experiment_name}."
    help_evaluate = "Evaluate CGP model perfomance such as MSE, Energy, Area, Delay, Depth, Gate count and CNN accuracy and loss. Weights are trained as they are defined by {experiment_name}."
    help_metacentrum = "Prepare file structure and a PBS file for training in Metacentrum. Dataset is generated according to {experiment_name}."

    for command, help in zip(experiment_commands, [help_train, help_metacentrum, help_evaluate, ""]):
        for experiment_name in experiment_names:
            experiment_parser = subparsers.add_parser(f"{experiment_name}:{command}", help=help.format(experiment_name=experiment_name))
            experiment_class = experiments_classes.get(experiment_name)

            if command != "fix-train-stats":
                experiment_parser.add_argument("model_name", help="Name of the model to optimize")
                experiment_parser.add_argument("model_path", help="Path to the model to optimize")

            experiment_parser.add_argument("cgp_binary_path", help="Path to the CGP binary")

            cgp_group = experiment_parser.add_argument_group("Cartesian Genetic Programming")
            CGPConfiguration.get_cgp_arguments(cgp_group)

            experiment_group = experiment_parser.add_argument_group("Experiment")
            if command in ["train", "train-pbs"]:
                experiment_group.add_argument("--experiment-env", help="Create a new isolated environment", nargs="?", default="experiment_results")
            elif command in ["evaluate"]:
                experiment_group.add_argument("--file", nargs='+', help="List of file paths to evaluate", required=False)
            elif command in ["fix-train-stats"]:
                experiment_group.add_argument("--experiment-root", nargs='?', help="Experiment root", required=False)

            experiment_group = experiment_class.get_argument_parser(experiment_group)
            experiment_group = experiment_class.get_base_argument_parser(experiment_group)

            pbs_group = experiment_parser.add_argument_group("PBS Metacentrum")
            if "pbs" in command:
                pbs_group = Experiment.get_train_pbs_argument_parser(pbs_group)                

            experiment_parser.set_defaults(factory=experiment_factories.get(experiment_name), experiment_name=experiment_name)

def register_commands(parser: argparse._SubParsersAction):
    subparsers = parser.add_subparsers(dest="command")
    _register_experiment_commands(subparsers, experiments_classes.keys())
    _register_model_commands(subparsers)

def dispatch(args):
    try:
        colon = args.command.index(":")
        experiment_name, command = args.command[:colon], args.command[colon+1:]
        print(experiment_name, command)
        if experiment_name == "model":
            raise ValueError(f"invalid experiment name {experiment_name}")

        if command == "train":
            return lambda: optimize_model(args)
        if command == "train-pbs":
            return lambda: optimize_prepare_model(args)
        if command == "evaluate":
            return lambda: evaluate_cgp_model(args)
        if command == "fix-train-stats":
            return lambda: fix_train_statistics(args)
        else:
            raise ValueError(f"unknown commmand {args.command}")
    except ValueError as e:
        if args.command == "model:train":
            return lambda: train_model(args.model_name, args.model_path, args.base)
        elif args.command == "model:evaluate":
            return lambda: evaluate_base_model(args.model_name, args.model_path, args.weights)
        elif args.command == "model:mobilenet_v2":
            return lambda: evaluate_base_model("mobilenet_v2", None, None)
        elif args.command == "model:quantize":
            return lambda: quantize_model(args.model_name, args.model_path, args.new_path)
        elif args.command == "model:debug":
            return lambda: debug_model(args.model_name, args.model_path)
        else:
            print("Invalid command. Use --help for usage information.")
            print(e)