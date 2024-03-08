import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns;

sns.set(style="ticks")
sns.set_context("paper")  # talk, notebook, paper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set font family, bold, and font size
font = {'size': 16}  # font = {'family':'normal', 'weight':'normal', 'size': 12}
matplotlib.rc('font', **font)
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'


cfg = {
  'x_label': 'Step',
  'y_label': 'Return',
  'show': False,
  'imgType': 'pdf',
  'ci': 'se',
  'runs': 20,
  'loc': 'best'
}


def nchain_best():
  label_list = ['DQN', 'Noisy-Net', 'Bootstrapped DQN', 'Adam LMCDQN', 'ULMCDQN', 'FG-LMCDQN', 'FG-ULMCDQN']
  color_list = ['tab:olive', 'tab:cyan', 'tab:purple', 'tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:brown']
  x_mean = np.array([25, 50, 75, 100])

  # Averaged over 20 runs
  y_mean = {
    'DQN': np.array([9.40, 5.52, 5.19, 5.05]),
    'FG-ULMCDQN': np.array([10, 9.85, 9.80, 9.80]),
    'FG-LMCDQN': np.array([9.50, 8.75, 8.46, 8.07]),
    'ULMCDQN': np.array([9.63, 9.73, 7.90, 8.02]),
    'Adam LMCDQN': np.array([8.55, 7.31, 6.08, 6.04]),
    'Noisy-Net': np.array([6.51, 1.95, 0.98, 0.90]),
    'Bootstrapped DQN': np.array([9.70, 7.56, 6.36, 3.57]),
    'LSVI-PHE': np.array([9.86, 9.86, 9.72, 9.50])
  }
  # 95% confidence interval
  y_ci = {
    'DQN': [0.92, 2.06717, 0.77441, 0.39727],
    'FG-ULMCDQN': [0.0, 0.292402, 0.313395, 0.39727],
    'FG-LMCDQN': [0.54218, 1.44277, 1.08837, 0.985371],
    'ULMCDQN': [0.27169, 0.25092, 1.83684, 1.46313],
    'Adam LMCDQN': [1.33972, 1.79998, 2.17492, 1.79998],
    'Noisy-Net': [2.13386, 1.77605, 1.15115, 0.293002],
    'Bootstrapped DQN': [0.2093, 1.33842, 1.65444, 1.52639],
    'LSVI-PHE': [0.27169, 0.27169, 0.54218, 0.64878]
  }

  fig, ax = plt.subplots()
  for i in range(len(label_list)):
    label = label_list[i]
    plt.plot(x_mean, y_mean[label], linewidth=2, color=color_list[i], label=label, marker='o', markersize=7)
    plt.fill_between(x_mean, y_mean[label] - y_ci[label], y_mean[label] + y_ci[label], facecolor=color_list[i],
             alpha=0.3)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlabel('N', fontsize=18)
  ax.set_ylabel('Return', fontsize=18)
  plt.tick_params(axis='both', which='major', labelsize=14)
  ax.set_xlim(20, 105)
  ax.set_ylim(0, None)
  ax.set_xticks(ticks=[25, 50, 75, 100])
  # Set legend
  ax.legend(loc=cfg['loc'], frameon=False, fontsize=14)
  # Adjust figure layout
  fig.tight_layout()
  # Save and show
  image_path = f'./figures/nchain_best.{cfg["imgType"]}'
  ax.get_figure().savefig(image_path)
  if cfg['show']:
    plt.show()
  plt.clf()  # clear figure
  plt.cla()  # clear axis
  plt.close()  # close window

if __name__ == "__main__":
  nchain_best()