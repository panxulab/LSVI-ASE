import os
import math
import numpy as np
from scipy.stats import bootstrap
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info
from utils.helper import set_one_thread


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-50:].mean(skipna=False) if mode=='Train' else result['Return'][-10:].mean(skipna=False),
    'Return (max)': result['Return'].max(skipna=False) if mode=='Train' else result['Return'].max(skipna=False)
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train', ci=90, method='percentile'):
  CI = bootstrap((result['Return (mean)'].values.tolist(),), np.mean, confidence_level=ci/100, method=method).confidence_interval
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=False),
    'Return (se)': result['Return (mean)'].sem(ddof=0),
    'Return (bootstrap_mean)': (CI.high + CI.low) / 2,
    f'Return (ci={ci})': (CI.high - CI.low) / 2,
    'Return (max)': result['Return (max)'].max(skipna=False)
  }
  return result_dict

cfg = {
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Return',
  'hue_label': 'Agent',
  'show': False,
  'imgType': 'png',
  'estimator': 'mean',
  'ci': ('ci', 95),
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'best',
  'sweep_keys': ['env/cfg/n', 'optimizer/name', 'optimizer/kwargs/noise_scale', 'optimizer/kwargs/gamma', 'optimizer/kwargs/lr', 'optimizer/kwargs/a', 'agent/update_num', 'agent/is_double', 'agent/eps_start'],
  'sort_by': ['Return (mean)', 'Return (max)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  set_one_thread()
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  # plotter.plot_results(mode='Test', indexes='all')


if __name__ == "__main__":
  runs = 20
  for exp in ['nchain_aULMC_v3_25']:
    analyze(exp, runs=runs)