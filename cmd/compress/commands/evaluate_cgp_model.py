import os
from string import Template
import pandas as pd
import seaborn as sns
import commands.datastore as store
import experiments.manager as experiments
from pathlib import Path
from experiments.single_channel.experiment import SingleChannelExperiment
from parse import parse
from commands.factory.experiment import create_all_experiment, create_experiment
from tqdm import tqdm
from functools import partial
from cgp.cgp_adapter import CGP

def evaluate_cgp_model(args):
    """
    Evaluates the CGP model for all experiments.

    Args:
        args: The arguments for creating and configuring the experiments.
    """    
    for experiment in create_all_experiment(args):
        experiment.config.set_start_run(args.start_run)
        experiment.config.set_start_generation(args.start_generation)
        experiment = experiment.get_result_eval_env()
        experiment.get_model_metrics_from_statistics()

def evaluate_model(root=r"C:\Users\Majo\source\repos\TorchCompresser\data_store\mobilenet\features_18_0_features_18_0_mse_0.0_256_31_batch_1", run=2):
    """
    Evaluates a specific CGP model.

    Args:
        root (str): The root directory of the model data.
        run (int): The run number to evaluate.
    """    
    root = Path(root)
    cgp = CGP("C:\\Users\\Majo\\source\\repos\\TorchCompresser\\out\\build\\x64-release\\cgp\\CGP.exe")
    df = pd.read_csv(
        root / f"train_statistics/fitness/statistics.{run}.csv",
        names=["run", "generation", "timestamp", "error", "qenergy", "energy", "area", "qdelay", "delay", "depth", "gate_count", "chromosome"])
    df.dropna(inplace=True, subset="chromosome")
    print(df)
    solution = df.iloc[-1]
    print(solution)
    with open("temp_chromosome.txt", "w") as f:
        f.write(solution["chromosome"] + "\n")
    cgp.evaluate_chromosomes("gate.stats.txt", train_weights=root / "train.data", config_path=root / "train_cgp.config", chromosome_file="temp_chromosome.txt", output_statistics="stats_temp_chromosome.txt", output_weights="weight_temp_chromosome.txt", gate_parameters_file=root / "gate_parameters.txt")     

def evaluate_model_metrics_pbs(
                        experiment: str = "mobilenet",
                        model_name: str = "mobilenet_v2",
                        model_path: str = "data_store/models/mobilenet_v2/mobilenet_v2.state_dict.pth",
                        time_limit: str = "24:00:00",
                        job_dir: str = "/storage/$server/home/$username/cgp_workspace/mobilenet_large_experiment/planners/batch_10_19",
                        dataset: str = None,
                        template_pbs_file: str = r"./cmd/compress/commands/pbs/model_metrics_job.sh",
                        data_dir: str = "/storage/$server/home/$username/cgp_workspace/mobilenet_large_experiment/mobilenet_preparation_batch_10_19",
                        results_folder: str = "/storage/$server/home/$username/cgp_workspace/mobilenet_large_experiment/results",
                        cgp_folder: str = "cgp_cpp_project",
                        cpu=16,
                        mem="64gb",
                        scratch_capacity="500gb",
                        modulo=1,
                        modulo_group=None,
                        batch_size=2048,
                        num_proc=1,
                        num_workers=14,
                        stats_format="statistics.{run}.csv.zip",
                        experiment_wildcard="*256_31",
                        **kwargs
                        ):
    """
    Evaluates model metrics using PBS.

    Args:
        experiment (str): The name of the experiment.
        model_name (str): The name of the model.
        model_path (str): The path to the model state dictionary.
        time_limit (str): The time limit for the job.
        job_dir (str): The directory for the job on remote.
        dataset (str): The dataset to be used.
        template_pbs_file (str): The path to the PBS template file.
        data_dir (str): The directory for data storage on remote.
        results_folder (str): The folder for storing results.
        cgp_folder (str): The folder containing the CGP project on remote achine.
        cpu (int): The number of CPUs to be used.
        mem (str): The memory allocation for the job.
        scratch_capacity (str): The scratch disk capacity.
        modulo (int): The modulo value for job grouping.
        modulo_group (Optional[int]): The specific modulo group to process.
        batch_size (int): The batch size for processing.
        num_proc (int): The number of processes.
        num_workers (int): The number of workers.
        stats_format (str): The format for statistics files.
        experiment_wildcard (str): The wildcard pattern for experiment selection.
        **kwargs: Additional keyword arguments.
    """    
    modulo_groups = [modulo_group] if modulo_group is not None else range(int(modulo))
    for modulo_group in modulo_groups:
        job_name = f"{experiment}_{model_name}_{modulo_group}_{modulo}"
        template_data = {
            "machine": f"select=1:ncpus={cpu}:ompthreads={cpu}:mem={mem}:scratch_ssd={scratch_capacity}",
            "model_name": model_name,
            "model_path": model_path,
            "time_limit": time_limit,
            "job_name": f"model_eval_{job_name}",
            "job_dir": job_dir,
            "server": os.environ.get("pbs_server"),
            "username": os.environ.get("pbs_username"),
            "hf_token": os.environ.get("huggingface"),
            "experiment_wildcard": experiment_wildcard,
            "stats_format": stats_format,
            "num_workers": num_workers,
            "num_proc": num_proc,
            "batch_size": batch_size,
            "data_dir": data_dir,
            "result_dir": results_folder,
            "cwd": "compress_py",
            "cgp_cpp_project": cgp_folder,
            "cgp_binary_src": "bin/cgp",
            "cgp_binary": "cgp",
            "experiment": experiment,
            "error_t": "uint64_t",
            "cflags": " ".join(["-D_DISABLE_ROW_COL_STATS", "-D_DEPTH_DISABLED"]),
            "dataset":  dataset,
            "modulo":  modulo,
            "modulo_group":  modulo_group,
        }

        pbs_file = f"{job_name}.pbs.sh"
        with open(template_pbs_file, "r") as template_pbs_f, open(pbs_file, "w", newline="\n") as pbs_f:
            copy_mode = False
            for line in template_pbs_f:
                if not copy_mode and line.strip() == "# PYTHON TEMPLATE END":
                    copy_mode = True
                    continue

                if copy_mode:
                    pbs_f.write(line)
                else:
                    # keep substituting until it is done
                    changed = True
                    while changed:
                        old_line = line
                        template = Template(line)
                        line = template.safe_substitute(template_data)
                        changed = line != old_line
                    pbs_f.write(line)
        print("saved pbs file to: " + str(pbs_file))    

columns_names = ["run", "generation", "timestamp", "error", "quantized_energy", "energy", "area", "quantized_delay", "delay", "depth", "gate_count", "chromosome"]

def sample(top: int, f: str):
    """
    Samples the top entries from a CSV file.

    Args:
        top (int): The number of top entries to sample.
        f (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """    
    df = pd.read_csv(f)
    if "error" in df.columns.values:
        return pd.concat([df[:-1].sample(n=top-1), df.tail(n=1)])
    else:
        df = pd.read_csv(f, names=columns_names)
        return pd.concat([df[:-1].sample(n=top-1), df.tail(n=1)])

def pick_top(top: int, f: str):
    """
    Picks the top entries from a CSV file.

    Args:
        top (int): The number of top entries to pick.
        f (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the top entries.
    """    
    prev_chunk = pd.DataFrame()
    last_chunk = pd.DataFrame()
    
    chunk = next(pd.read_csv(f, chunksize=2*top), None)
    names = None
    
    if chunk is None:
        raise ValueError(f"dataset is empty for {f}")
    
    if "error" not in chunk.columns.values:
        names = columns_names
    
    for chunk in pd.read_csv(f, chunksize=2*top, names=names):
        prev_chunk = last_chunk
        last_chunk = chunk
    
    df = pd.concat([prev_chunk, last_chunk])
    return df[-top:]

def evaluate_model_metrics(args):
    """
    Evaluates model metrics for experiments.

    Args:
        args: The arguments for creating and configuring the experiments.
    """    
    experiment_list = args.experiment
    only_weights = args.only_weights
    experiment = create_experiment(args, prepare=False)
    data_store = store.Datastore()
    data_store.init_experiment_path(experiment)
    df_factory = pd.read_csv if args.top is None else partial(pick_top, args.top)
    original_top = args.top
    kwargs = vars(args)
    del kwargs["top"]

    if not isinstance(experiment, experiments.MultiExperiment):
        experiment_list = [experiment]

    for case in experiment_list:
        sub_experiments = experiment.get_experiments_with_glob(case) if isinstance(experiment, experiments.MultiExperiment) else [experiment]
        for x in sub_experiments:
            if kwargs.get("rename", False):
                print(f"skipping {x.get_name(depth=1)} because it was renamed")
                continue
            for run in (args.runs or x.get_number_of_train_statistic_file(fmt=args.statistics_file_format)):
                destination = data_store.derive_from_experiment(experiment) / "model_metrics" / (f"{args.dataset or 'default'}.{args.split or 'test'}." + (x.get_name(depth=1) + f".{run}.csv"))
                if destination.exists():
                    print(f"skipping {x.get_name(depth=1)} run {run}")
                    continue
                print(f"evaluating {x.get_name(depth=1)} run {run}")
                file = x.get_train_statistics(runs=run, fmt=args.statistics_file_format)[0]
                df = df_factory(file)

                df.drop(columns="depth", inplace=True, errors="ignore")
                df = df[~(
                    df["error"].duplicated(keep="last") & df["quantized_energy"].duplicated(keep="last") & 
                    df["energy"].duplicated(keep="last") & df["quantized_delay"].duplicated(keep="last") & 
                    df["delay"].duplicated(keep="last") & df["area"].duplicated(keep="last") &
                    df["gate_count"].duplicated(keep="last") & df["chromosome"].duplicated(keep="last"))]   


                df = df.loc[~df["chromosome"].isna(), :].copy()
                df["Top-1"] = None
                df["Top-5"] = None
                df["Loss"] = None        
                
                top = original_top + 1 if original_top else len(df.index)
                chromosomes_file = data_store.derive_from_experiment(x) / f"chromosomes.{run}.txt"
                stats_file = data_store.derive_from_experiment(x) / "evaluate_statistics" /  f"statistics.{run}.csv"
                weights_file = data_store.derive_from_experiment(x) / "all_weights" / (f"weights.{run}." + "{run}.txt")
                gate_statistics = data_store.derive_from_experiment(x) / "gate_statistics" / (f"statistics.{run}." + "{run}.txt")
                weights_file.unlink(missing_ok=True)
                
                stats_file.parent.mkdir(exist_ok=True, parents=True)
                weights_file.parent.mkdir(exist_ok=True, parents=True)
                gate_statistics.parent.mkdir(exist_ok=True, parents=True)
                
                with open(chromosomes_file, "w") as f:
                    for _, row in df.iterrows():
                        f.write(row["chromosome"] + "\n")
                
                x.evaluate_chromosomes(chromosomes_file, stats_file, weights_file, gate_statistics)        
                
                if only_weights:
                    continue
                
                def run_identifier_iterator():
                    for i in range(1, top, 1):
                        current_path = Path(str(weights_file).format(run=i))
                        
                        if not current_path.exists():
                            break
                        
                        yield i                   
                
                def weight_iterator():
                    for i in run_identifier_iterator():
                        current_path = Path(str(weights_file).format(run=i))
                        yield x.get_weights(current_path)
                
                cached_top_k, cached_loss = None, None
                top_1 = []; top_5 = []; losses = []; runs_id = [];
                fitness_values = ["error", "quantized_energy", "energy", "area", "quantized_delay", "delay", "depth", "gate_count", "chromosome"]
                with tqdm(zip(weight_iterator(), df.iterrows(), pd.read_csv(stats_file).iterrows(), run_identifier_iterator()), unit="Record", total=len(df.index), leave=True) as records:
                    for (weights, plans), (index, row), (eval_index, eval_row), run_id in records:
                        df.loc[index, fitness_values] = eval_row[fitness_values]
                        runs_id.append(run_id)
                        print("start error:", row["error"], "new error:", eval_row["error"])  
                        if False and eval_row["error"] == 0:
                            if cached_top_k is None:
                                cached_top_k, cached_loss = x.get_reference_model_metrics(cache=False, batch_size=None, top=[1, 5])
                            top_1.append(cached_top_k[1])
                            top_5.append(cached_top_k[5])
                            losses.append(cached_loss)
                        else:          
                            print("error:", eval_row["error"])   
                            if False and eval_row["error"] == 0:
                                top_1.append(cached_top_k[1])
                                top_5.append(cached_top_k[5])
                                losses.append(cached_loss)
                            else:
                                model = x._model_adapter.inject_weights(weights, plans)
                                top_k, loss = model.evaluate(top=[1, 5], **kwargs)
                                top_1.append(top_k[1])
                                top_5.append(top_k[5])
                                losses.append(loss)
                df["Top-1"] = top_1
                df["Top-5"] = top_5
                df["Loss"] = losses
                df["Run ID"] = runs_id
                destination.parent.mkdir(exist_ok=True, parents=True)                                                                                             
                df.to_csv(destination, index=False)