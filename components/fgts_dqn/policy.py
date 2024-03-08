from copy import deepcopy
from typing import Any, Dict, Optional, Union, Callable

import torch
import numpy as np

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.policy.base import _nstep_return


class FGTSLangevinDQNPolicy(DQNPolicy):
  """Implementation of FGTS Langevin DQN.
  
  :param torch.nn.Module model: a model following the rules in
    :class:`~tianshou.policy.BasePolicy`. (s -> logits)
  :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
  :param float discount_factor: in [0, 1].
  :param int estimation_step: the number of steps to look ahead. Default to 1.
  :param int target_update_freq: the target network update frequency (0 if
    you do not use the target network). Default to 0.
  :param bool reward_normalization: normalize the reward to Normal(0, 1).
    Default to False.
  :param bool is_double: use double dqn. Default to True.
  :param bool clip_loss_grad: clip the gradient of the loss in accordance
    with nature14236; this amounts to using the Huber loss instead of
    the MSE loss. Default to False.
  :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
    optimizer in each policy.update(). Default to None (no lr_scheduler).

  .. seealso::

    Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
    explanation.
  """
  def __init__(
    self,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    discount_factor: float = 0.99,
    estimation_step: int = 1,
    target_update_freq: int = 0,
    reward_normalization: bool = False,
    is_double: bool = False,
    clip_loss_grad: bool = False,
    **kwargs: Any,
  ) -> None:
    super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)
    self.num_ensemble = model.num_ensemble

  def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray, head: int) -> torch.Tensor:
    batch = buffer[indices]  # batch.obs_next: s_{t+n}
    if self._target:
      target_q = self(batch, model="model_old", input="obs_next", head=head).logits
    else:
      target_q = self(batch, model="model", input="obs_next", head=head).logits
    return target_q.max(dim=1)[0]

  def process_fn(
    self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray, head: int
  ) -> Batch:
    # Compute the n-step noise return for Langevin Q-learning targets.
    batch = self.compute_nstep_return(
        batch, buffer, indices, self._target_q, self._gamma, self._n_step,
        self._rew_norm, head=head
    )
    return batch

  def forward(
    self,
    batch: Batch = None,
    state: Optional[Union[dict, Batch, np.ndarray]] = None,
    model: str = "model",
    input: str = "obs",
    head: int = None,
    **kwargs: Any
  ) -> Batch:
    """Compute action over the given batch data for a given or randomly chosen head.

    :return: A :class:`~tianshou.data.Batch` which has 3 keys:

      * ``act`` the action.
      * ``logits`` the network's raw output.
      * ``state`` the hidden state.

    .. seealso::

      Please refer to :meth:`~tianshou.policy.DQNPolicy.forward` for
      more detailed explanation.
    """
    # Randomly choose one head if not assigned
    if head is None:
      head = np.random.randint(low=0, high=self.num_ensemble)
    model = getattr(self, model)
    obs = batch[input]
    obs_next = obs.obs if hasattr(obs, "obs") else obs
    logits, hidden = model(obs_next, state=state, info=batch.info, head=head)
    q = self.compute_q_value(logits, getattr(obs, "mask", None))
    if not hasattr(self, "max_action_num"):
      self.max_action_num = q.shape[1]
    act = to_numpy(q.max(dim=1)[1])
    return Batch(logits=logits, act=act, state=hidden)
        
  def compute_nstep_return(
    self,
    batch: Batch,
    buffer: ReplayBuffer,
    indice: np.ndarray,
    target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
    gamma: float = 0.99,
    n_step: int = 1,
    rew_norm: bool = False,
    head: int = None
  ) -> torch.Tensor:
    r"""Compute n-step return give the head.
    """
    assert not rew_norm, \
      "Reward normalization in computing n-step returns is unsupported now."
    rew = buffer.rew
    bsz = len(indice)
    indices = [indice]
    for _ in range(n_step - 1):
      indices.append(buffer.next(indices[-1]))
    indices = np.stack(indices)
    # terminal indicates buffer indexes nstep after 'indice',
    # and are truncated at the end of each episode
    terminal = indices[-1]
    with torch.no_grad():
      target_q_torch = target_q_fn(buffer, terminal, head=head)  # (bsz, ?)
    target_q = to_numpy(target_q_torch.reshape(bsz, -1))
    target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
    end_flag = buffer.done.copy()
    end_flag[buffer.unfinished_index()] = True
    target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
    batch.returns = to_torch_as(target_q, target_q_torch)
    if hasattr(batch, "weight"):  # prio buffer update
      batch.weight = to_torch_as(batch.weight, target_q_torch)
    return batch
  
  def learn(self, batch: Batch, head: int, **kwargs: Any) -> Dict[str, float]:
    if self._target and self._iter % self._freq == 0:
      self.sync_weight()
    self.optim.zero_grad()
    weight = batch.pop("weight", 1.0)
    q = self(batch, head=head).logits
    q = q[np.arange(len(q)), batch.act]
    returns = to_torch_as(batch.returns.flatten(), q)
    td_error = returns - q

    if self._clip_loss_grad:
      y = q.reshape(-1, 1)
      t = returns.reshape(-1, 1)
      loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
    else:
      loss = (td_error.pow(2) * weight).mean()

    batch.weight = td_error  # prio-buffer
    loss.backward()
    self.optim.step()
    self._iter += 1
    return {"loss": loss.item()}

  def update(self,
    sample_size: int = None, buffer: Optional[ReplayBuffer] = None, head: int = None, 
    **kwargs: Any) -> Dict[str, Any]:
    """Update the policy network and replay buffer.

    It includes 3 function steps: process_fn, learn, and post_process_fn. In
    addition, this function will change the value of ``self.updating``: it will be
    False before this function and will be True when executing :meth:`update`.
    Please refer to :ref:`policy_state` for more detailed explanation.

    :param int sample_size: 0 means it will extract all the data from the buffer,
      otherwise it will sample a batch with given sample_size.
    :param ReplayBuffer buffer: the corresponding replay buffer.

    :return: A dict, including the data needed to be logged (e.g., loss) from
      ``policy.learn()``.
    """
    if buffer is None:
      return {}
    batch, indices = buffer.sample(sample_size)
    self.updating = True
    # Randomly choose one head to update
    if head is None:
      head = np.random.randint(low=0, high=self.num_ensemble)
    batch = self.process_fn(batch, buffer, indices, head=head)
    result = self.learn(batch, head=head, **kwargs)
    self.post_process_fn(batch, buffer, indices)
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()
    self.updating = False
    return result