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
# manager.py: Manage argparse CLI arguments.

import argparse
import os
from commands.debug_model import debug_model
import experiments.manager as experiments
from commands.optimize_prepare_model import optimize_prepare_model
from commands.fix_train_stats import fix_train_statistics
from commands.optimize_model import optimize_model
from commands.evaluate_cgp_model import evaluate_cgp_model, evaluate_model_metrics, evaluate_model_metrics_pbs
from commands.train_model import train_model
from commands.evaluate_model import evaluate_base_model
from commands.evaluate_model_sensitivity import model_sensitivity
from commands.quantize_model import quantize_model
from cgp.cgp_configuration import CGPConfiguration
from commands.datastore import Datastore
from typing import List


experiment_commands = ["train", "train-pbs", "evaluate", "fix-train-stats", "model-metrics"]
required_cgp = {
    "train": True,
    "train-pbs": False,
    "evaluate": True,
    "fix-train-stats": True,
    "model-metrics": True
}

def _register_model_commands(subparsers: argparse._SubParsersAction):
    """
    Registers model-related commands to the argument parser.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers action object to register commands to.
    """    
    # model:train
    train_parser = subparsers.add_parser("model:train", help="Train a model")
    train_parser.add_argument("model_name", help="Name of the model to train")
    train_parser.add_argument("-m", "--model-path", help="Path where trained model will be saved")
    train_parser.add_argument("-b", "--base", type=str, help="Path to the baseline model")

    # model:evaluate
    evaluate_parser = subparsers.add_parser("model:evaluate", help="Evaluate a model")
    evaluate_parser.add_argument("model_name", help="Name of the model to evaluate")
    evaluate_parser.add_argument("-m", "--model-path", help="Path to the model to evaluate")
    evaluate_parser.add_argument("--weights", nargs="?", type=str, help="", default=[], required=False)
    evaluate_parser.add_argument("--batch-size", type=int, default=None, help="Batch size for evaluation")
    evaluate_parser.add_argument("--top", nargs="+", type=int, default=[1, 5], help="Top-k accuracy to compute")
    evaluate_parser.add_argument("-l", "--include-loss", action="store_true", help="Whether to include loss in evaluation")
    evaluate_parser.add_argument("--show-top-k", type=int, default=2, help="Number of top-k accuracies to display")
    evaluate_parser.add_argument("-a", "--archive", action="store_true", help="Archive model in datastore")
    evaluate_parser.add_argument("--num-workers", type=int, default=None, help="Worker count for data loader")
    evaluate_parser.add_argument("--num-proc", type=int, default=None, help="Proccesor count for dataset")
    evaluate_parser.add_argument("--dataset", help="Dataset to use", type=str)
    evaluate_parser.add_argument("--split", help="Split to use", type=str)

    # model:evaluate
    sensitivity_parser = subparsers.add_parser("model:sensitivity", help="Evaluate a model sensitivity")
    sensitivity_parser.add_argument("model_name", help="Name of the model to evaluate")
    sensitivity_parser.add_argument("-e", "--error-type", nargs="+", type=str, default=["inner", "outter", "all"], help="Weight error types")
    sensitivity_parser.add_argument("-m", "--model-path", help="Path to the model to evaluate")
    sensitivity_parser.add_argument("--batch-size", type=int, default=None, help="Batch size for evaluation")
    sensitivity_parser.add_argument("--top", nargs="+", type=int, default=[1, 5], help="Top-k accuracy to compute")
    sensitivity_parser.add_argument("-l", "--include-loss", action="store_true", help="Whether to include loss in evaluation")
    sensitivity_parser.add_argument("--show-top-k", type=int, default=2, help="Number of top-k accuracies to display")
    sensitivity_parser.add_argument("--num-workers", type=int, default=None, help="Worker count for data loader")
    sensitivity_parser.add_argument("--num-proc", type=int, default=None, help="Proccesor count for dataset")

    # model:quantize
    quantize_parser = subparsers.add_parser("model:quantize", help="Quantize a model")
    quantize_parser.add_argument("model_name", help="Name of the model to quantize")
    quantize_parser.add_argument("-m", "--model-path", help="Path where trained model is saved")
    quantize_parser.add_argument("new_path", help="Path of the new quantized model where it will be stored")    

    # model:debug
    debug_parser = subparsers.add_parser("model:debug", help="Debug a model")
    debug_parser.add_argument("model_name", help="Name of the model to quantize")
    debug_parser.add_argument("-m", "--model_path", help="Path where trained model is saved", required=False, default=None) 

    parser = subparsers.add_parser("model-metrics-pbs", help="Model evaluation on PBS")
    parser.add_argument("--experiment", type=str, default="mobilenet", help="Experiment name")
    parser.add_argument("--model-name", type=str, default="mobilenet_v2", help="Model name")
    parser.add_argument("--model-path", type=str, default="data_store/models/mobilenet_v2/mobilenet_v2.state_dict.pth", help="Path to the model")
    parser.add_argument("--time-limit", type=str, default="24:00:00", help="Time limit")
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument("--template-pbs-file", type=str, default=r"C:\Users\Majo\source\repos\TorchCompresser\cmd\compress\commands\pbs\model_metrics_job.sh", help="Template PBS file path")
    parser.add_argument("--data-dir", type=str, default="/storage/$server/home/$username/cgp_workspace/mobilenet_large_experiment/mobilenet_preparation_batch_10_19", help="Data directory")
    parser.add_argument("--job-dir", type=str, default="/storage/$server/home/$username/cgp_workspace/mobilenet_large_experiment/planners/batch_10_19", help="OpenPBS job directory")
    parser.add_argument("--results-folder", type=str, default="/storage/$server/home/$username/cgp_workspace/mobilenet_large_experiment/results", help="Results folder")
    parser.add_argument("--cgp-folder", type=str, default="cgp_cpp_project", help="CGP folder")
    parser.add_argument("--cpu", type=int, default=16, help="CPU")
    parser.add_argument("--mem", type=str, default="64gb", help="Memory")
    parser.add_argument("--scratch-capacity", type=str, default="500gb", help="Scratch capacity")
    parser.add_argument("--modulo-group", type=int, default=None, help="Modulo group")
    parser.add_argument("--modulo", type=int, default=None, help="Modulo")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes")
    parser.add_argument("--num-workers", type=int, default=14, help="Number of workers")
    parser.add_argument("--stats-format", type=str, default="statistics.{run}.csv.zip", help="Statistics format")
    parser.add_argument("--experiment-wildcard", type=str, default="*256_31", help="Experiment wildcard")

def _register_experiment_commands(subparsers: argparse._SubParsersAction, experiment_names: List[str]):
    """
    Registers experiment-related commands to the argument parser.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers action object to register commands to.
        experiment_names (List[str]): A list of experiment names to register.
    """    
    help_train = "Train a new CGP model to infer mising convolution weights from CNN model. Weights are trained as they are defined by {experiment_name}."
    help_evaluate = "Evaluate CGP model perfomance such as MSE, Energy, Area, Delay, Depth, Gate count and CNN accuracy and loss. Weights are trained as they are defined by {experiment_name}."
    help_metacentrum = "Prepare file structure and a PBS file for training in Metacentrum. Dataset is generated according to {experiment_name}."

    for command, help in zip(experiment_commands, [help_train, help_metacentrum, help_evaluate, "", ""]):
        for experiment_name in experiment_names:
            experiment_parser = subparsers.add_parser(f"{experiment_name}:{command}", help=help.format(experiment_name=experiment_name))
            experiment_parser.add_argument("--cgp", help="Path to the CGP binary", type=str, default=os.environ.get("cgp", None), required=("cgp" not in os.environ and required_cgp[command]))
            experiment_parser.add_argument("--experiment", help="Specific sub-experiments", type=str, nargs="+", default=[])
            CGPConfiguration.get_cgp_arguments(experiment_parser.add_argument_group("Cartesian Genetic Programming"))
            
            if command != "fix-train-stats":
                experiment_parser.add_argument("model_name", help="Name of the model to optimize")
                experiment_parser.add_argument("model_path", help="Path to the model to optimize")            
            
            experiment_group = experiment_parser.add_argument_group("Experiment")
            if command in ["train", "train-pbs"]:
                experiment_group.add_argument("--experiment-env", help="Create a new isolated environment", nargs="?", default="experiment_results")

            if command == "model-metrics":
                experiment_group.add_argument("--runs", help="Specific runs to evaluate", nargs="+", default=None)
                experiment_group.add_argument("--top", help="Evaluate only the best", type=int, default=None)
                experiment_group.add_argument("--batch-size", help="Batch size", type=int, default=None)
                experiment_group.add_argument("-s", "--statistics-file-format", help="Specify statistics filename", type=str, default=None)
                experiment_parser.add_argument("--experiment-path", help="Path to the experiment", type=str, default=str(Datastore().derive(experiment_name)))
                experiment_group.add_argument("-w", "--only-weights", help="Infer weights only", action="store_true")
                experiment_group.add_argument("--dataset", help="Dataset to use", type=str)
                experiment_group.add_argument("--split", help="Split to use", type=str)
                experiment_group.add_argument("--num-workers", type=int, default=None, help="Worker count for data loader")
                experiment_group.add_argument("--num-proc", type=int, default=None, help="Proccesor count for dataset")
                experiment_group.add_argument("-l", "--include-loss", action="store_true", help="Whether to include loss in evaluation")
                
                if experiment_name == "mobilenet":
                    experiment_group.add_argument("--rename", action="store_true", help="Whether to only rename old experiment format")
            else:
                experiment_parser.add_argument("--experiment-path", help="Path to the experiment", type=str, default=os.environ.get("experiments_root", "cmd/compress/experiments/"))

            experiments.get_experiment_arguments(experiment_name, experiment_group)
            experiments.get_base_argument_parser(experiment_group)

            pbs_group = experiment_parser.add_argument_group("PBS Metacentrum")
            if "pbs" in command:
                pbs_group = experiments.get_train_pbs_argument_parser(pbs_group)                

            experiment_parser.set_defaults(factory=experiments.get_experiment_factory(experiment_name), experiment_name=experiment_name)

def register_commands(parser: argparse._SubParsersAction):
    """
    Registers all commands to the argument parser.

    Args:
        parser (argparse._SubParsersAction): The argument parser to register commands to.
    """    
    subparsers = parser.add_subparsers(dest="command")
    _register_experiment_commands(subparsers, experiments.experiments_classes.keys())
    _register_model_commands(subparsers)

def dispatch(args):
    """
    Dispatches the command based on the parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A callable function to execute the command.

    Raises:
        ValueError: If the command is invalid or unknown.
    """    
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
        if command == "model-metrics":
            return lambda: evaluate_model_metrics(args)
        else:
            raise ValueError(f"unknown commmand {args.command}")
    except ValueError as e:
        if args.command == "model:train":
            return lambda: train_model(args.model_name, args.model_path, args.base)
        elif args.command == "model:evaluate":
            return lambda: evaluate_base_model(**vars(args))
        elif args.command == "model:sensitivity":
            return lambda: model_sensitivity(**vars(args))
        elif args.command == "model:quantize":
            return lambda: quantize_model(args.model_name, args.model_path, args.new_path)
        elif args.command == "model:debug":
            return lambda: debug_model(args.model_name, args.model_path)
        elif args.command == "model-metrics-pbs":
            return lambda: evaluate_model_metrics_pbs(**vars(args))
        else:
            print("Invalid command. Use --help for usage information.")
            print(e)