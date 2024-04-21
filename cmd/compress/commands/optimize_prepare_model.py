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
        cgp = CGP(args.cgp_binary_path)
        model = BaseAdapter.from_base_model(args.model_name, args.model_path) if "model_name" in args else None

        if model is not None:
            model.load()
            model.eval()

        if "experiment_root" not in args:
            config = CGPConfiguration(f"cmd/compress/experiments/{args.experiment_name}/config.cgp")
            config.parse_arguments(args)
        else:
            config = args.experiment_root

        experiment = factory(config=config, model_adapter=model, cgp=cgp, args=args)

        if isinstance(experiment, MultiExperiment):
            for x in experiment.experiments.values() if "experiment_root" not in args else experiment.get_experiments():
                yield x
        else:
            yield experiment

def optimize_prepare_model(args):
    for experiment in prepare_experiment(args):
        if not experiment.config.has_start_run():
            experiment.config.set_start_run(args.start_run)
        if not experiment.config.has_start_generation():
            experiment.config.set_start_generation(args.start_generation)
        experiment = experiment.get_isolated_train_env(args.experiment_env, relative_paths=True)
        cpu = args.cpu
        if not experiment.config.has_population_max() and cpu is None:
            raise ValueError("population_max or cpu argument is needed in order to create pbs job")
        elif not experiment.config.has_population_max() and cpu is not None:
            experiment.config.set_population_max(args.cpu)
        elif experiment.config.has_population_max() and cpu is None:
            cpu = int(experiment.config.get_population_max())
        elif experiment.config.has_population_max() and cpu is not None:
            print("warn: population_max and cpu is set; leaving it as it is")
        
        experiment.setup_pbs_train_job(
            args.time_limit,
            args.template_pbs_file,
            experiments_folder=args.experiments_folder,
            results_folder=args.results_folder,
            cgp_folder=args.cgp_folder,
            cpu=cpu,
            mem=args.mem,
            scratch_capacity=args.scratch_capacity)
                                       