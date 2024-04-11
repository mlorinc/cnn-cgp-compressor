from typing import List, Optional, Generator
from itertools import zip_longest
import torch
from models.mobilenet_v2 import MobileNetV2
from models.base import BaseModel
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment
from experiments.multi_experiment import MultiExperiment
import importlib
import argparse
import pandas as pd

def _get_model(model_name: str, model_path: Optional[str] = None) -> BaseModel:
    return importlib.import_module(f"models.{model_name}").init(model_path)

def _get_experiment(experiment_base_name: str, config: CGPConfiguration, model: BaseModel, cgp: CGP, dtype=torch.int8, **args) -> Experiment:
    return importlib.import_module(f"experiments.{experiment_base_name}.experiment").init(config, model, cgp, dtype=dtype, **args)

def train_model(model_name: str, model_path: str, base: str):
    print(f"Training model: {model_name}")
    model: BaseModel = _get_model(model_name, model_path)
    if base:
        model.load(base)
        print(f"Loaded from {base}")

    model.fit()
    model.save(model_path)

def evaluate_model(model_name: str, model_path: str):
    print(f"Evaluating model: {model_name}")
    model: BaseModel = _get_model(model_name, model_path)
    if model_name != MobileNetV2.name:
        model = model.load(model_path)
    acc, loss = model.evaluate(max_batches=None)
        
    print(f"acc: {acc:.12f}%, loss {loss:.12f}")

def quantize_model(model_name, model_path, new_path):
    print(f"Quantizing model: {model_name} and saving as {new_path}")
    model: BaseModel = _get_model(model_name, model_path)
    model.load(model_path, quantized=False)
    model.quantize(new_path)
    model.save(new_path)

def prepare_experiment(model_name: str, model_path: str, cgp_binary_path: str, experiment_names: List[str], **args) -> Generator[Experiment, None, None]:
    for experiment_name in experiment_names:
        model = _get_model(model_name, model_path)
        model = model.load(model_path)
        model.eval()

        cgp = CGP(cgp_binary_path)
        config = CGPConfiguration(f"cmd/compress/experiments/{experiment_name}/config.cgp")
        experiment = _get_experiment(experiment_name, config, model, cgp, **args)

        if isinstance(experiment, MultiExperiment):
            for x in experiment.experiments.values():
                yield x
        else:
            yield experiment

def optimize_model(cgp_binary_path: str, args):
    for experiment in prepare_experiment(args.model_name, args.model_path, cgp_binary_path, args.experiment_names, **args):
        experiment = experiment.setup_isolated_train_environment(args.experiment_env)
        experiment.train(start_run=args.start_run, start_generation=args.start_generation)

def optimize_prepare_model(cgp_binary_path: str, args):
    for experiment in prepare_experiment(args.model_name, args.model_path, cgp_binary_path, args.experiment_names, **args):
        experiment.setup_isolated_train_environment(args.experiment_env)

def evaluate_cgp_model(cgp_binary_path: str, args):
    for experiment in prepare_experiment(args.model_name, args.model_path, cgp_binary_path, args.experiment_names, **args):
        experiment = experiment.setup_eval_environment()
        experiment.evaluate_runs()

def main():
    parser = argparse.ArgumentParser(description="Model Training, Evaluation, Quantization, and Optimization")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # model:train
    train_parser = subparsers.add_parser("model:train", help="Train a model")
    train_parser.add_argument("model_name", help="Name of the model to train")
    train_parser.add_argument("model_path", help="Path where trained model will be saved")
    train_parser.add_argument("-b", "--base", type=str, help="Path to the baseline model")

    # model:evaluate
    evaluate_parser = subparsers.add_parser("model:evaluate", help="Evaluate a model")
    evaluate_parser.add_argument("model_name", help="Name of the model to evaluate")
    evaluate_parser.add_argument("model_path", help="Path to the model to evaluate")

    # model:quantize
    quantize_parser = subparsers.add_parser("model:quantize", help="Quantize a model")
    quantize_parser.add_argument("model_name", help="Name of the model to quantize")
    quantize_parser.add_argument("model_path", help="Path where trained model is saved")
    quantize_parser.add_argument("new_path", help="Path of the new quantized model where it will be stored")

    # cgp:optimize
    optimize_parser = subparsers.add_parser("cgp:optimize", help="Optimize a model")
    optimize_parser.add_argument("model_name", help="Name of the model to optimize")
    optimize_parser.add_argument("model_path", help="Path to the model to optimize")
    optimize_parser.add_argument("cgp_binary_path", help="Path to the CGP binary")
    optimize_parser.add_argument("experiment_name", nargs="+", help="Experiment to evaluate")
    optimize_parser.add_argument("--start_run", help="Continue in experimentation from given start run", nargs="?", default=None)
    optimize_parser.add_argument("--start_generation", help="Continue in evolution from given start generation", nargs="?", default=None)
    optimize_parser.add_argument("--experiment_env", help="Isolated experiment environment", nargs="?", default="cmd/compress/experiment_results")

    # cgp:optimize-prepare
    optimize_preapre_parser = subparsers.add_parser("cgp:optimize-prepare", help="Prepare experiments for optimisations")
    optimize_preapre_parser.add_argument("model_name", help="Name of the model to optimize")
    optimize_preapre_parser.add_argument("model_path", help="Path to the model to optimize")
    optimize_preapre_parser.add_argument("cgp_binary_path", help="Path to the CGP binary")
    optimize_preapre_parser.add_argument("experiment_name", nargs="+", help="Experiment to evaluate")
    optimize_preapre_parser.add_argument("--experiment_env", help="Isolated experiment environment", nargs="?", default="cmd/compress/experiment_results")

    # cgp:optimize
    evaluate_cgp_parser = subparsers.add_parser("cgp:evaluate", help="Evalaute a model")
    evaluate_cgp_parser.add_argument("model_name", help="Name of the model to evaluate")
    evaluate_cgp_parser.add_argument("model_path", help="Path to the model to evaluate")
    evaluate_cgp_parser.add_argument("cgp_binary_path", help="Path to the CGP binary")
    evaluate_cgp_parser.add_argument("experiment_name", nargs="+", help="Experiment to evaluate")
    evaluate_cgp_parser.add_argument("--run", nargs='+', type=int, help="List of runs to evaluate", required=False)
    evaluate_cgp_parser.add_argument("--file", nargs='+', help="List of file paths to evaluate", required=False)
    evaluate_cgp_parser.add_argument("--experiment_env", help="Isolated experiment environment", nargs="?", default="cmd/compress/experiment_results")

    args = parser.parse_args()

    if args.command == "model:train":
        train_model(args.model_name, args.model_path, args.base)
    elif args.command == "model:evaluate":
        evaluate_model(args.model_name, args.model_path)
    elif args.command == "model:quantize":
        quantize_model(args.model_name, args.model_path, args.new_path)
    elif args.command == "cgp:optimize":
        optimize_model(args.model_name, args.model_path, args.cgp_binary_path, args.experiment_name, args)
    elif args.command == "cgp:optimize-prepare":
        optimize_prepare_model(args.model_name, args.model_path, args.cgp_binary_path, args.experiment_name, args)
    elif args.command == "cgp:evaluate":
        evaluate_cgp_model(args.model_name, args.model_path, args.cgp_binary_path, args.experiment_name, args)
    else:
        print("Invalid command. Use --help for usage information.")

if __name__ == "__main__":
    main()
