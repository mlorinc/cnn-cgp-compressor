from commands.optimize_prepare_model import prepare_experiment

def optimize_model(args):
    for experiment in prepare_experiment(args):
        # experiment.config.set_start_run(args.start_run)
        # experiment.config.set_start_generation(args.start_generation)
        experiment = experiment.setup_isolated_train_environment(args.experiment_env)
        experiment.train(start_run=args.start_run, start_generation=args.start_generation)
