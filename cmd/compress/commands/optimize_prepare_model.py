import torch
from commands.evaluate_model import get_model_adapter
from cgp.cgp_adapter import CGP
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment
from experiments.multi_experiment import MultiExperiment
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from typing import Generator

def get_experiment(config: CGPConfiguration, model_adapter: ModelAdapter, cgp: CGP, args, dtype=torch.int8) -> Experiment:
    return args.factory(config=config, model_adapter=model_adapter, cgp=cgp, args=args)

def prepare_experiment(args) -> Generator[Experiment, None, None]:
    factories = args.factories if "factories" in args else args.factory
    factories = factories if isinstance(factories, list) else [factories]

    for factory in factories:
        model = BaseAdapter.from_base_model(args.model_name, args.model_path)
        model.load()
        model.eval()

        cgp = CGP(args.cgp_binary_path)
        config = CGPConfiguration(f"cmd/compress/experiments/{args.experiment_name}/config.cgp")
        config.parse_arguments(args)
        experiment = factory(config=config, model_adapter=model, cgp=cgp, args=args)

        if isinstance(experiment, MultiExperiment):
            for x in experiment.experiments.values():
                yield x
        else:
            yield experiment

def optimize_prepare_model(args):
    for experiment in prepare_experiment(args):
        experiment.config.set_start_run(args.start_run)
        experiment.config.set_start_generation(args.start_generation)
        experiment = experiment.setup_isolated_train_environment(args.experiment_env, relative_paths=True)
        experiment.setup_pbs_train_job(
            args.time_limit,
            args.template_pbs_file,
            experiments_folder=args.experiments_folder,
            results_folder=args.results_folder,
            cgp_folder=args.cgp_folder,
            cpu=args.cpu,
            mem=args.mem,
            scratch_capacity=args.scratch_capacity)
                                       