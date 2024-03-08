import gym
from gym.wrappers.time_limit import TimeLimit

from envs.wrapper import *


def make_atari_env(task, agent, seed, training_num, test_num, **kwargs):
  """Wrapper function for Atari env.
  If EnvPool is installed, it will automatically switch to EnvPool's Atari env.
  :return: a tuple of (single env, training envs, test envs).
  """
  if agent == 'HyperDQN':
    env = wrap_deepmind(task, **kwargs)
    train_envs = SubprocVectorEnv([
      lambda: wrap_deepmind(task, episode_life=True, clip_rewards=True, **kwargs)
      for _ in range(training_num)
    ])
    test_envs = ShmemVectorEnv([
      lambda: wrap_deepmind(task, episode_life=False, clip_rewards=False, **kwargs)
      for _ in range(test_num)
    ])
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
  elif envpool is not None:
    if kwargs.get("scale", 0):
      warnings.warn(
        "EnvPool does not include ScaledFloatFrame wrapper, "
        "please set `x = x / 255.0` inside CNN network's forward function."
      )
    # parameters convertion
    train_envs = env = envpool.make_gym(
      task.replace("NoFrameskip-v4", "-v5"),
      num_envs=training_num,
      seed=seed,
      episodic_life=True,
      reward_clip=True,
      stack_num=kwargs.get("frame_stack", 4),
    )
    test_envs = envpool.make_gym(
      task.replace("NoFrameskip-v4", "-v5"),
      num_envs=test_num,
      seed=seed,
      episodic_life=False,
      reward_clip=False,
      stack_num=kwargs.get("frame_stack", 4),
    )
  else:
    warnings.warn(
      "Recommend using envpool (pip install envpool) "
      "to run Atari games more efficiently."
    )
    env = wrap_deepmind(task, **kwargs)
    train_envs = ShmemVectorEnv([
      lambda: wrap_deepmind(task, episode_life=True, clip_rewards=True, **kwargs)
      for _ in range(training_num)
    ])
    test_envs = ShmemVectorEnv([
      lambda: wrap_deepmind(task, episode_life=False, clip_rewards=False, **kwargs)
      for _ in range(test_num)
    ])
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
  return env, train_envs, test_envs


def make_exploration_env(task, agent, seed, training_num, test_num, **kwargs):
  import gym_exploration
  if task in ['DiabolicalCombLock-v0', 'NChain-v1']:
    env = gym.make(task, **kwargs)
    train_envs = SubprocVectorEnv([lambda: gym.make(task, **kwargs) for _ in range(training_num)])
    test_envs = ShmemVectorEnv([lambda: gym.make(task, **kwargs) for _ in range(test_num)])
  elif task == 'SparseMountainCar-v0':
    if kwargs['max_episode_steps'] > 0: # Set max episode steps
      env = TimeLimit(gym.make(task).unwrapped, kwargs['max_episode_steps'])
      train_envs = SubprocVectorEnv([lambda: TimeLimit(gym.make(task).unwrapped, kwargs['max_episode_steps']) for _ in range(training_num)])
      test_envs = ShmemVectorEnv([lambda: TimeLimit(gym.make(task).unwrapped, kwargs['max_episode_steps']) for _ in range(test_num)])
    else:
      env = gym.make(task)
      train_envs = SubprocVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
      test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
  env.seed(seed)
  train_envs.seed(seed)
  test_envs.seed(seed)
  return env, train_envs, test_envs


def make_gym_env(task, agent, seed, training_num, test_num, **kwargs):
  assert envpool is not None, 'Please install envpool.'
  '''
  Default max_episode_steps in envpool: https://github.com/sail-sg/envpool/blob/main/envpool/classic_control/registration.py
  '''
  train_envs = env = envpool.make_gym(
    task,
    num_envs=training_num,
    seed=seed,
    max_episode_steps=kwargs['max_episode_steps']
  )
  test_envs = envpool.make_gym(
    task,
    num_envs=test_num,
    seed=seed,
    max_episode_steps=kwargs['max_episode_steps']
  )
  return env, train_envs, test_envs