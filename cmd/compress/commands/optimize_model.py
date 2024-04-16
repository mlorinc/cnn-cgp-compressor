from commands.optimize_prepare_model import prepare_experiment
from experiments.experiment import MissingChromosomeError

def optimize_model(args):
    for experiment in prepare_experiment(args):
        # experiment.config.set_start_run(args.start_run)
        # experiment.config.set_start_generation(args.start_generation)
        experiment = experiment.setup_isolated_train_environment(args.experiment_env)
        last_run = experiment.get_number_of_experiment_results()
        
        if last_run == experiment.config.get_number_of_runs():
            print("skipping " + experiment.get_name())
            continue
        
        try:
            if last_run != 0:
                experiment = experiment.setup_resumed_train_environment()
        except MissingChromosomeError:
            pass
            
        experiment.train_cgp()
