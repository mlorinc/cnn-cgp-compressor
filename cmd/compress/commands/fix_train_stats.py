from commands.optimize_prepare_model import prepare_experiment

def fix_train_statistics(args):
    for experiment in prepare_experiment(args):
        experiment = experiment.setup_statistics_fix_environment()
        print("fixing experiment: " + experiment.get_name())
        experiment.evaluate_missing_statistics()
