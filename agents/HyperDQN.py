from envs.env import *
from utils.helper import *
from agents.BaseAgent import *

from components.hyper_dqn.network import HyperDQNNet
from components.hyper_dqn.policy import HyperDQNPolicy
from components.hyper_dqn.trainer import offpolicy_trainer
from components.hyper_dqn.buffer import VectorReplayBuffer


class HyperDQN(BaseAgent):
  '''
  Implementation of HyperDQN
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Create Q network
    self.net = self.createNN()
    # Set optimizer
    self.optimizer = getattr(torch.optim, self.cfg['optimizer']['name'])(self.net.parameters(), **self.cfg['optimizer']['kwargs'])
    # Set replay buffer: `save_last_obs` and `stack_num` can be removed when you have enough RAM
    self.buffer = VectorReplayBuffer(
      size = self.cfg['buffer_size'],
      buffer_num = self.cfg['env']['train_num'],
      ignore_obs_next = True,
      save_only_last_obs = self.save_only_last_obs,
      stack_num = self.cfg['frames_stack'],
      mask_prob = 0.0,
      num_ensemble = 1,
      noise_dim = self.cfg['agent']['z_size']
    )
    # Define policy
    self.policy = HyperDQNPolicy(
      model = self.net,
      optim = self.optimizer,
      discount_factor = self.discount,
      estimation_step = self.cfg['n_step'],
      target_update_freq = self.cfg['target_update_steps'],
      reward_normalization = False,
      is_double = True,
      noise_scale = self.cfg['agent']['noise_scale'],
      l2_norm = self.cfg['agent']['l2_norm'],
      num_train_iter = self.cfg['agent']['num_train_iter']
    )
    # Set Collectors
    self.collectors =  {
      'Train': Collector(self.policy, self.envs['Train'], self.buffer, exploration_noise=True),
      'Test': Collector(self.policy, self.envs['Test'], exploration_noise=True)
    }
    self.logger.info("Observations shape: {}".format(self.state_shape))
    self.logger.info("Actions shape: {}".format(self.action_shape))

  def createNN(self):
    NN = HyperDQNNet(
      *self.state_shape,
      action_shape = self.action_shape, 
      z_size = self.cfg['agent']['z_size'], 
      bias_coef = self.cfg['agent']['bias_coef'],
      prior_scale = self.cfg['agent']['prior_scale'], 
      posterior_scale = self.cfg['agent']['posterior_scale'],
      prior_mean = self.cfg['agent']['prior_mean'], 
      prior_std = self.cfg['agent']['prior_scale'],
      device = self.device
    )
    return NN.to(self.device)

  def run_steps(self):
    # Test train_collector and start filling replay buffer
    self.collectors['Train'].collect(n_step=self.batch_size * self.cfg['env']['train_num'])
    # Trainer
    result = offpolicy_trainer(
      policy = self.policy,
      train_collector = self.collectors['Train'],
      test_collector = self.collectors['Test'],
      max_epoch = self.cfg['epoch'],
      step_per_epoch = self.cfg['step_per_epoch'],
      step_per_collect = self.cfg['step_per_collect'],
      episode_per_test = 5,
      batch_size = self.batch_size,
      update_per_step = self.cfg['update_per_step'],
      train_fn = self.train_fn,
      test_fn = self.train_fn,
      save_best_fn = self.save_model if self.cfg['save_model'] else None,
      logger = self.logger,
      verbose = True,
      # Set it to True to show speed, etc.
      show_progress = self.cfg['show_progress'],
      test_in_train = True,
      learning_start = self.cfg['agent']['learning_start']
    )
    for k, v in result.items():
      self.logger.info(f'{k}: {v}')

  def train_fn(self, epoch, env_step):
    eps = 1.0 if env_step <= self.cfg['agent']['learning_start'] else 0.0
    self.policy.set_eps(eps)