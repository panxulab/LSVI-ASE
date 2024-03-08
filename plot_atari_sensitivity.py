import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks"); sns.set_context("paper") # talk, notebook, paper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# Set font family, bold, and font size
font = {'size': 16} # font = {'family':'normal', 'weight':'normal', 'size': 12}
matplotlib.rc('font', **font)
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

from utils.helper import make_dir
from utils.plotter import read_file, get_total_combination, symmetric_ema, moving_average


class Plotter(object):
  def __init__(self, cfg):
    cfg.setdefault('ci', 'se')
    cfg.setdefault('rolling_score_window', -1)
    self.ci = cfg['ci']
    self.runs = cfg['runs']
    self.x_label = cfg['x_label']
    self.y_label = cfg['y_label']
    self.imgType = cfg['imgType']
    self.rolling_score_window = cfg['rolling_score_window']
    make_dir('./figures/')

  def get_result(self, exp, cfg_idx, mode):
    '''
    Given exp and config index, get the results
    '''
    total_combination = get_total_combination(exp)
    result_list = []
    for _ in range(self.runs):
      result_file = f'./logs/{exp}/{cfg_idx}/result_{mode}.feather'
      # If result file exist, read and merge
      result = read_file(result_file)
      if result is not None:
        # Add config index as a column
        result['Config Index'] = cfg_idx
        result_list.append(result)
      cfg_idx += total_combination
    # Get x's and y's in form of numpy arries
    xs, ys = [], []
    for result in result_list:
      xs.append(result[self.x_label].to_numpy())
      ys.append(result[self.y_label].to_numpy())
    # Moving average
    if self.rolling_score_window > 0:
      for i in range(len(xs)):
        ys[i] = moving_average(ys[i], self.rolling_score_window)
    # Do symetric EMA to get new x's and y's
    low  = max(x[0] for x in xs)
    high = min(x[-1] for x in xs)
    n = min(len(x) for x in xs)
    for i in range(len(xs)):
      new_x, new_y, _ = symmetric_ema(xs[i], ys[i], low, high, n)
      result_list[i] = result_list[i][:n]
      result_list[i].loc[:, self.x_label] = new_x
      result_list[i].loc[:, self.y_label] = new_y
    # Convert to numpy array
    ys = []
    for result in result_list:
      ys.append(result[self.y_label].to_numpy())
    # Put all results in a dataframe
    ys = np.array(ys)
    x_mean = result_list[0][self.x_label].to_numpy() * 4 / 1e6
    runs = len(result_list)
    x = np.tile(x_mean, runs)
    y = ys.reshape((-1))
    result_df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
    return result_df

cfg = {
  'x_label': 'Step',
  'y_label': 'Return',
  'imgType': 'pdf',
  # 'imgType': 'pdf',
  'estimator': 'mean',
  #'ci': 'se',
  # 'estimator': 'median',
  'ci': ('ci', 95),
  'rolling_score_window': 10,
  'runs': 5,
  'loc': 'best'
}


def vary_eta(env, runs=5):
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  label_list = ['$\\eta=1$', '$\\eta=10^{-1}$', '$\\eta=10^{-2}$', '$\\eta=10^{-3}$', '$\\eta=10^{-4}$', '$\\eta=10^{-5}$']
  color_list = ['tab:purple', 'tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:cyan']
  exp_list = ['alien_aULMC_fg','alien_aULMC_fg','alien_aULMC_fg','alien_aULMC_fg','alien_aULMC_fg','alien_aULMC_fg']
  index_list = [1, 2, 3, 4, 5, 6]

  fig, ax = plt.subplots()
  for j in range(len(label_list)):
    agent, color, exp, config_idx = label_list[j], color_list[j], exp_list[j], index_list[j]
    print(f'[{env}]: Plot Test results: {config_idx}')
    result_df = plotter.get_result(exp, config_idx, 'Test')
    # Plot
    sns.lineplot(
      data=result_df, x='x', y='y',
      estimator=cfg['estimator'],
      errorbar=cfg['ci'], err_kws={'alpha':0.3},
      linewidth=3, color=color, label=agent
    )
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # Set x and y axis
  ax.set_xlabel("Frame (millions)", fontsize=18)
  ax.set_ylabel('Return', fontsize=18)
  plt.tick_params(axis='both', which='major', labelsize=14)
  # Set legend
  ax.legend(loc=cfg['loc'], frameon=False, fontsize=14)
  # Save and show
  fig.tight_layout()
  image_path = f'./figures/{env}_eta.{cfg["imgType"]}'
  ax.get_figure().savefig(image_path)
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window




def vary_noise(env, runs=5):
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  label_list = ['$\\beta_k=10^{20}$', '$\\beta_k=10^{18}$', '$\\beta_k=10^{16}$', '$\\beta_k=10^{14}$', '$\\beta_k=10^{12}$', '$\\beta_k=10^{10}$', '$\\beta_k=10^{8}$']
  color_list = ['tab:purple', 'tab:pink', 'tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:olive', 'tab:cyan']
  exp_list = ['qbert_beta', 'qbert_beta', 'atari8_lmc1', 'atari8_lmc1', 'atari8_lmc1', 'qbert_beta', 'qbert_beta']
  index_list = [1, 2, 103, 135, 167, 3, 4]

  fig, ax = plt.subplots()
  for j in range(len(label_list)):
    agent, color, exp, config_idx = label_list[j], color_list[j], exp_list[j], index_list[j]
    print(f'[{env}]: Plot Test results: {config_idx}')
    result_df = plotter.get_result(exp, config_idx, 'Test')
    # Plot
    sns.lineplot(
      data=result_df, x='x', y='y',
      estimator=cfg['estimator'],
      errorbar=cfg['ci'], err_kws={'alpha':0.3},
      linewidth=3, color=color, label=agent
    )
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # Set x and y axis
  ax.set_xlabel("Frame (millions)", fontsize=18)
  ax.set_ylabel('Return', fontsize=18)
  plt.tick_params(axis='both', which='major', labelsize=14)
  # Set legend
  ax.legend(loc=cfg['loc'], frameon=False, fontsize=14, ncol=2)
  # Save and show
  fig.tight_layout()
  image_path = f'./figures/{env}_noise.{cfg["imgType"]}'
  ax.get_figure().savefig(image_path)
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window


def vary_a(env, runs=5):
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  label_list = ['a=10', 'a=1', 'a=0.1', 'a=0.01', 'a=0.001', 'a=0']
  color_list = ['tab:purple', 'tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:cyan']
  exp_list = ['qbert_a', 'atari8_lmc1', 'atari8_lmc1', 'atari8_lmc1', 'qbert_a', 'qbert_a2']
  index_list = [1, 71, 167, 263, 2, 3]

  fig, ax = plt.subplots()
  for j in range(len(label_list)):
    agent, color, exp, config_idx = label_list[j], color_list[j], exp_list[j], index_list[j]
    print(f'[{env}]: Plot Test results: {config_idx}')
    result_df = plotter.get_result(exp, config_idx, 'Test')
    # Plot
    sns.lineplot(
      data=result_df, x='x', y='y',
      estimator=cfg['estimator'],
      errorbar=cfg['ci'], err_kws={'alpha':0.3},
      linewidth=3, color=color, label=agent
    )
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # Set x and y axis
  ax.set_xlabel("Frame (millions)", fontsize=18)
  ax.set_ylabel('Return', fontsize=18)
  plt.tick_params(axis='both', which='major', labelsize=14)
  # Set legend
  ax.legend(loc=cfg['loc'], frameon=False, fontsize=14)
  # Save and show
  fig.tight_layout()
  image_path = f'./figures/{env}_a.{cfg["imgType"]}'
  ax.get_figure().savefig(image_path)
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window


if __name__ == "__main__":
  # vary_noise('qbert')
  #vary_a('qbert')
  vary_eta('alien')