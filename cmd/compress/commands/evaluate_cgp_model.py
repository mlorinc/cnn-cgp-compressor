from commands.optimize_prepare_model import prepare_experiment

def evaluate_cgp_model(args):
    for experiment in prepare_experiment(args):
        experiment.config.set_start_run(args.start_run)
        experiment.config.set_start_generation(args.start_generation)
        experiment = experiment.setup_eval_environment()
        experiment.get_model_metrics()


