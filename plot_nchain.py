import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns;

sns.set(style="ticks");
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
# plt.rcParams['axes.xmargin'] = 0
# plt.rcParams['axes.ymargin'] = 0


cfg = {
    'x_label': 'Step',
    'y_label': 'Return',
    'show': False,
    # 'imgType': 'png',
    'imgType': 'pdf',
    'ci': 'se',
    'runs': 20,
    'loc': 'best'
}


def nchain_best():
    label_list = ['DQN', 'Noisy-Net', 'Bootstrapped DQN', 'Adam LMCDQN', 'ULMCDQN', 'FG-LMCDQN', 'FG-ULMCDQN']
    color_list = ['tab:olive', 'tab:cyan', 'tab:purple', 'tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:brown']
    x_mean = np.array([25, 50, 75, 100])

    # Data with 20 runs
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
    y_ci = {
        'DQN': np.array([0.44, 0.99, 0.37, 0.19])*2.093,
        'FG-ULMCDQN': np.array([0, 0.14, 0.15, 0.19])*2.093,
        'FG-LMCDQN': np.array([0.26, 0.69, 0.52, 0.47])*2.093,
        'ULMCDQN': np.array([0.13, 0.12, 0.88, 0.70])*2.093,
        'Adam LMCDQN': np.array([0.64, 0.86, 1.04, 0.86])*2.093,
        'Noisy-Net': np.array([1.02, 0.85, 0.55, 0.14])*2.093,
        'Bootstrapped DQN': np.array([0.10, 0.64, 0.79, 0.73])*2.093,
        'LSVI-PHE': np.array([0.13, 0.13, 0.26, 0.31])*2.093
    }

    fig, ax = plt.subplots()
    for i in range(len(label_list)):
        label = label_list[i]
        # Plot
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


def nchain_update_num():
    x = np.array([1, 4, 16, 32])
    # old data with 10 runs
    y = np.array([5.05, 5.40, 5.61, 6.27])  # averaged top 10 performance for N=100
    y_ci = np.array([0.27, 0.35, 0.17, 0.22])
    # y = np.array([6.54, 8.22, 6.54, 7.92]) # top 1 performance for N=100
    # y = np.array([4.95, 5.00, 4.92, 5.37])
    fig, ax = plt.subplots()
    # Plot
    plt.plot(x, y, linewidth=2, color='b')
    plt.fill_between(x, y - y_ci, y + y_ci, facecolor='b', alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("The Update Number", fontsize=18)
    ax.set_ylabel('Return', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Adjust figure layout
    fig.tight_layout()
    # Save and show
    image_path = f'./figures/nchain_update_num.{cfg["imgType"]}'
    ax.get_figure().savefig(image_path)
    if cfg['show']:
        plt.show()
    plt.clf()  # clear figure
    plt.cla()  # clear axis
    plt.close()  # close window


if __name__ == "__main__":
    nchain_best()