# Parallel Recursive LSTM

This repository contains the code used for the experiments in the paper.

## Environment

The experiments were run on Windows Subsystem for Linux (WSL) using Python 3.12, JAX, and CUDA 12. GPU acceleration is recommended for the main experimental runs.

## Installation
```shell
pip install .
```

## Experiments
The commands below reproduce the main experimental runs used in the paper. Unless otherwise stated, results are reported from the network scores printed to standard output.

### Main Results
The main results in Table 2 are produced with:
```shell
python experiments/main.py --task-name even_pairs modular_arithmetic parity_check cycle_navigation stack_manipulation reverse_string modular_arithmetic_brackets solve_equation duplicate_string missing_duplicate_string odds_first binary_addition binary_multiplication compute_sqrt bucket_sort --model-name parallel_recursive -M 460 --testing-lengths "n+40" --training-steps 1000000 --log-frequency 100000
```

### Training-time results
The results in Figure 3 are produced with:
```shell
python ./experiments/main.py --task-name even_pairs modular_arithmetic parity_check cycle_navigation stack_manipulation reverse_string modular_arithmetic_brackets solve_equation duplicate_string missing_duplicate_string odds_first binary_addition binary_multiplication compute_sqrt bucket_sort --model-name parallel_recursive lstm transformer_encoder -M 1 --training-steps 40000 --log-frequency 200 --save-dir ./saved/
```

This command writes the corresponding CSV-style log files to `./saved/`, with filenames of the form:

```<task-name>-<model-name>-training.dat```

## Inference-time and memory-scaling results
The results in Figure 4 are produced with:
```shell
python experiments/main.py --task-name parity_check --model-name parallel_recursive lstm transformer_encoder -M 500 --testing-lengths "max(1,5*(n-1))" --training-steps 1 --test-batches 1 --test-batch-size 1024 --save-dir ./saved/ --plugins speed
```
This command writes the corresponding speed logs to `./saved/`, with filenames of the form:
```parity_check-<model-name>-speed-1024.dat```

## Ablation results
The ablation results in Table 3 are produced with the following three commands.

First, the 0R and 2R variants of PR-LSTM are produced with:
```shell
python experiments/main.py --task-name even_pairs modular_arithmetic parity_check cycle_navigation stack_manipulation reverse_string modular_arithmetic_brackets solve_equation duplicate_string missing_duplicate_string odds_first binary_addition binary_multiplication compute_sqrt bucket_sort --model-name parallel_recursive --cell-name lstm --num-layers 1 3 -M 460 --testing-lengths n+40 --training-steps 1000000 --log-frequency 100000
```
Second, the 0R, 1R, and 2R variants of PR-RNN are produced with:
```shell
python experiments/main.py --task-name even_pairs modular_arithmetic parity_check cycle_navigation stack_manipulation reverse_string modular_arithmetic_brackets solve_equation duplicate_string missing_duplicate_string odds_first binary_addition binary_multiplication compute_sqrt bucket_sort --model-name parallel_recursive --cell-name rnn --num-layers 1 2 3 -M 460 --testing-lengths n+40 --training-steps 1000000 --log-frequency 100000
```
Finally, the parameter-matched PR-LSTM variant is produced with:
```shell
python experiments/main.py --task-name even_pairs modular_arithmetic parity_check cycle_navigation stack_manipulation reverse_string modular_arithmetic_brackets solve_equation duplicate_string missing_duplicate_string odds_first binary_addition binary_multiplication compute_sqrt bucket_sort --model-name parallel_recursive --hidden-size 137 -M 460 --testing-lengths n+40 --training-steps 1000000 --log-frequency 100000
```
and fetching the logged network scores in the standard output.