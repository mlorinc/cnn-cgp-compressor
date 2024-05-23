(Disclaimer: the following work is still in progress; citations will be inserted later)

# Automated Compression of Neural Network Weights

The following work was done as parth of my master's thesis and to share
code in conventional way I have decided to create a repository which was
used to obrain experimental results.

## Thesis Abstract

Convolutional Neural Networks have revolutionised computer vision field since their
introduction. By replacing weights with convolution filters containing trainable weights,
CNNs significantly reduced memory usage. However, this reduction came at the cost of in-
creased computational resource requirements, as convolution operations are more computation-
intensive. Despite this, memory usage remains more energy-intensive than computation.

This thesis explores whether it is possible to avoid loading weights from memory and
instead functionally calculate them, thereby saving energy. To test this hypothesis, a novel
weight compression algorithm was developed using Cartesian Genetic Programming. This
algorithm searches for the most optimal weight compression function, aiming to enhance
energy efficiency without compromising the functionality of the neural network.

Experiments conducted on the LeNet-5 and MobileNetV2 architectures demonstrated
that the algorithm could effectively reduce energy consumption while maintaining high
model accuracy. The results showed that certain layers could benefit from weight compu-
tation, validating the potential for energy-efficient neural network implementations.


## What is Included

This repository contains only the implementation code. The experiments need to be generated and will be published soon. Data cleaning is required to avoid cluttering the repository. In the meantime, Jupyter Notebooks are available in the cmd/compress directory, each covering one experiment.

## Install

To install all dependencies, the Conda environment has been exported to conda-requirements.txt with pip requirements listed in requirements.txt. If this installation method fails, the project depends on the following packages: PyTorch, pandas, numpy, seaborn, parse, tqdm, scipy, scikit-posthocs, and Python 3.11.5.

If computations are to be performed on Metacentrum, the scripts operate in $HOME/cgp_workspace, where the CGP project must be copied to $HOME/cgp_workspace/cgp_cpp_project, and experiments must be copied to $HOME/cgp_workspace/experiments.

For local development, CGP can be compiled using Visual Studio or the provided Makefile. Subsequently, the .env file should be set, for instance: cgp=C:\\Users\\Majo\\source\\repos\\TorchCompresser\\out\\build\\x64-release\\cgp\\CGP.exe and datastore=C:\\Users\\Majo\\source\\repos\\TorchCompresser\\data_store. If MobileNetV2 is being tested, huggingface=<token> should also be set.

# Usage

To train in simple way, for example: 

```sh
python ./cmd/compress/compress.py layer_bypass:train qat_quantized_lenet qat.quantized_lenet.pth --population-max 8 --experiment-env ./local_experiments/ --patience 500000 --mse-thresholds 0 -e 0  --rows 256 --cols 31 --mutation-max 0.01
```

To generate PBS experiments, for example: 

```sh
python ./cmd/compress/compress.py le_selector:train-pbs qat_quantized_lenet qat.quantized_lenet.pth --time-limit 48:00:00 --template-pbs-file ./compress/experiments/job.pbs --population-max 8 --mem 500mb --experiment-env experiment-pbs --scratch-capacity 20gb -b 30 --patience 10000000 --mse-thresholds 0 --rows 256 --cols 31 --mutation-max 0.01 --multiplex
```

To obtain model metrics, for instance: 

```sh
python ./cmd/compress/compress.py le_selector:model-metrics qat_quantized_lenet qat.quantized_lenet.pth --experiment *_256_31  -s "statistics.{run}.csv.zip" --top 1 --dataset qmnist --split nist --num-workers 4 --num-proc 1 --batch-size 40 --include-loss --e-fitness SE
```

Please note that the flag -s might differ depending on how it was bundled on MetaCentrum using the experiment.sh script in the metacentrum folder. The script is used in the following way, assuming the working directory is in the experiment results that were batched into smaller batches.

```sh
experiment.sh merge <experiment_name without _batch_x> 31 all
```

It is important to note that models are loaded from data_store if their values were not specified with the option -m. The program has many more utilities, but these are the most frequently used.

## License

As for now, it is not allowed to use work for any purposes. The license will be published when text is published.
