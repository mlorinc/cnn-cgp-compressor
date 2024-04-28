from pathlib import Path
from cgp.cgp_configuration import CGPConfiguration
from experiments.experiment import Experiment
from experiments.composite.experiment import MultiExperiment
import models.adapters.model_adapter_factory as model_factory
from typing import Generator, Union
import os

def _get_config(experiment_path, experiment_name):
    if experiment_path.endswith(".cgp"):
        config = CGPConfiguration(experiment_path)
    else:
        files = os.listdir(experiment_path)
        config = next((f for f in files if f.endswith(".cgp")), None)
        if config:
            config = CGPConfiguration(os.path.join(experiment_path, config))
        else:
            path = Path(os.path.join(experiment_path, "config.cgp"))
            if path.exists():
                config = CGPConfiguration(path)
            else:
                path = Path(os.path.join(experiment_path, experiment_name, "config.cgp"))
                if path.exists():
                    config = CGPConfiguration(path)
                else: 
                    config = experiment_path
                    # raise ValueError(f"invalid: {experiment_path}")
    return config

def create_experiment(args, prepare=True) -> Experiment:
    factories = args.factories if "factories" in args else args.factory
    factories = factories if isinstance(factories, list) else [factories]

    for factory in factories:
        model = model_factory.create_adapter(args.model_name, args.model_path)

        if model is not None:
            model.eval()

        config = _get_config(args.experiment_path, args.experiment_name)
        if isinstance(config, CGPConfiguration):
            config.parse_arguments(vars(args))
        args = vars(args)
        if "cgp" in args:
            cgp = args["cgp"]
            del args["cgp"]
        else:
            cgp = None
        experiment = factory(config=config, model_adapter=model, cgp=cgp, args=args, prepare=prepare)

        return experiment
        
def create_all_experiment(args) -> Generator[Experiment, None, None]:
    experiment = create_experiment(args)

    if isinstance(experiment, MultiExperiment):
        for x in experiment.experiments.values():
            yield x
    else:
        yield experiment
        