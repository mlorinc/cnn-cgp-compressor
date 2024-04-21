from commands.optimize_prepare_model import prepare_experiment

def fix_train_statistics(args):
    for experiment in prepare_experiment(args):
        experiment = experiment.get_statistics_fix_env()
        print("fixing experiment: " + experiment.get_name())
        experiment.evaluate_chromosome_in_statistics()
