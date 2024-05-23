from commands.factory.experiment import create_all_experiment

def optimize_prepare_model(args):
    """
    Prepares the model for optimization by setting up the necessary environment and creating PBS job files for training.

    Args:
        args: Parsed command-line arguments containing the necessary parameters for the preparation process.

    Workflow:
        1. Creates all experiments based on the provided arguments.
        2. Configures the start run and start generation if not already set.
        3. Sets up an isolated training environment with relative paths.
        4. Ensures that either the population_max or cpu argument is provided and sets them accordingly.
        5. Sets up the PBS job for training with the specified parameters.
    """    
    for experiment in create_all_experiment(args):
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
                                       