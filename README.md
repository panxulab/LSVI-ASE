# More Efficient Randomized Exploration in RL with Approximate Sampling

This repository contains the source code for the paper titled "More Efficient Randomized Exploration for Reinforcement Learning via Approximate Sampling."

## Installation Requirements

- Python: >= 3.8
- [Tianshou](https://github.com/thu-ml/tianshou): ==0.4.10
- [Envpool](https://github.com/sail-sg/envpool): ==0.6.6
- Additional dependencies can be found in `requirements.txt`.

## Running Experiments

### Setting Up and Executing Experiments

Hyperparameters and grid search parameters are organized within a configuration file located in the `configs` folder. To initiate an experiment, select a configuration index to generate a corresponding dictionary. This dictionary defines the specific experiment setup. All outputs, including logs, are stored within the `logs` folder. For detailed instructions, refer to the provided source code.

To launch an experiment using the configuration file `atari8_fg_aULMC.json` with the index `1`, execute:

```python main.py --config_file ./configs/atari8_fg_aULMC.json --config_idx 1```

### Optional: Grid Search

To identify the total number of parameter combinations for a given configuration (for instance, `atari8_fg_aULMC.json`), run:

`python utils/sweeper.py`

This command outputs the total combinations:

`Number of total combinations in atari8_fg_aULMC.json: 1728`

To systematically explore each combination (indices 1 to 144), you could utilize a bash script:

```bash
for index in {1..144}
do
  python main.py --config_file ./configs/atari8_fg_aULMC.json --config_idx $index
done
```

For handling a large batch of experiments, [GNU Parallel](https://www.gnu.org/software/parallel/) is recommended for job scheduling:

```bash
parallel --eta --ungroup python main.py --config_file ./configs/atari8_fg_aULMC.json --config_idx {1} ::: $(seq 1 1728)
```

If conducting multiple runs for the same configuration index, increment the index by the total number of combinations. For instance, to perform 5 runs for index `1`:

```
for index in 1 1729 3457 5185 6912
do
  python main.py --config_file ./configs/atari8_fg_aULMC.json --config_idx $index
done
```

Alternatively, for simplicity:

```
parallel --eta --ungroup python main.py --config_file ./configs/atari8_fg_aULMC.json --config_idx {1} ::: $(seq 1 1728 8640)
```

### Optional: Analyzing Results

To analyze experiment outcomes, simply execute:

`python analysis.py`

This script identifies unfinished experiments by checking for missing result files, reports memory usage, and produces a histogram of memory utilization for the `logs/atari8_fg_aULMC/0` directory. It also generates CSV files summarizing the training and testing outcomes. For comprehensive details, see `analysis.py`. Additional analysis tools are available in `utils/plotter.py`.