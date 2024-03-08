# Efficient Randomized Exploration in RL with Approximate Sampling

This repository contains the source code for the paper titled "More Efficient Randomized Exploration for Reinforcement Learning via Approximate Sampling."

## Installation Requirements

- Python version 3.8 or higher
- [Tianshou](https://github.com/thu-ml/tianshou) library
- [Envpool](https://github.com/sail-sg/envpool) library
- Additional dependencies can be found in `requirements.txt`.

## Running Experiments

### Setting Up and Executing Experiments

Hyperparameters and grid search parameters are organized within a configuration file located in the `configs` folder. To initiate an experiment, select a configuration index to generate a corresponding dictionary. This dictionary defines the specific experiment setup. All outputs, including logs, are stored within the `logs` folder. For detailed instructions, refer to the provided source code.

To launch an experiment using the configuration file `qbert_fg_aulmc.json` with the index `1`, execute:

```python main.py --config_file ./configs/qbert_fg_aulmc.json --config_idx 1```

### Optional: Grid Search

To identify the total number of parameter combinations for a given configuration (for instance, `qbert_fg_aulmc.json`), run:

`python utils/sweeper.py`

This command outputs the total combinations:

`Number of total combinations in qbert_fg_aulmc.json: 144`

To systematically explore each combination (indices 1 to 144), you could utilize a bash script:

```bash
for index in {1..144}
do
  python main.py --config_file ./configs/qbert_fg_aulmc.json --config_idx $index
done
```

For handling a large batch of experiments, [GNU Parallel](https://www.gnu.org/software/parallel/) is recommended for job scheduling:

```bash
parallel --eta --ungroup python main.py --config_file ./configs/qbert_fg_aulmc.json --config_idx {1} ::: $(seq 1 144)
```

If conducting multiple runs for the same configuration index, increment the index by the total number of combinations. For instance, to perform 5 runs for index `1`:

```
for index in 1 145 289 433 577
do
  python main.py --config_file ./configs/qbert_fg_aulmc.json --config_idx $index
done
```

Alternatively, for simplicity:

```
parallel --eta --ungroup python main.py --config_file ./configs/qbert_fg_aulmc.json --config_idx {1} ::: $(seq 1 144 720)
```

### Optional: Analyzing Results

To analyze experiment outcomes, simply execute:

`python analysis.py`

This script identifies unfinished experiments by checking for missing result files, reports memory usage, and produces a histogram of memory utilization for the `logs/qbert_fg_aulmc/0` directory. It also generates CSV files summarizing the training and testing outcomes. For comprehensive details, see `analysis.py`. Additional analysis tools are available in `utils/plotter.py`.