from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from tianshou.data import ReplayBuffer, Batch, to_numpy, to_torch_as
from tianshou.policy import DQNPolicy


class FGPriorDQNPolicy(DQNPolicy):
  """Implementation of FGPrior DQN.
  
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
    update_num: int = 1,
    eta: float = 0.1,
    **kwargs: Any,
  ) -> None:
    super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)
    self.update_num = update_num
    self.eta = eta

  def forward(
          self,
          batch: Batch,
          state: Optional[Union[dict, Batch, np.ndarray]] = None,
          model: str = "model",
          input: str = "obs",
          **kwargs: Any,
  ) -> Batch:
      """Compute action over the given batch data.

      :return: A :class:`~tianshou.data.Batch` which has 3 keys:

          * ``act`` the action.
          * ``logits`` the network's raw output.
          * ``state`` the hidden state.

      .. seealso::

          Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
          more detailed explanation.
      """
      model = getattr(self, model)
      obs = batch[input]
      obs_next = obs.obs if hasattr(obs, "obs") else obs
      logits, hidden = model(obs_next, state=state, info=batch.info)
      q = self.compute_q_value(logits, getattr(obs, "mask", None))
      if not hasattr(self, "max_action_num"):
          self.max_action_num = q.shape[1]
      act = to_numpy(q.max(dim=1)[1])
      return Batch(logits=logits, act=act, state=hidden)

  def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
      if self._target and self._iter % self._freq == 0:
          self.sync_weight()
      self.optim.zero_grad()
      weight = batch.pop("weight", 1.0)
      q = self(batch).logits
      fg_prior = self(batch, input="obs_first").logits.max(dim=1)[0]
      q = q[np.arange(len(q)), batch.act]
      returns = to_torch_as(batch.returns.flatten(), q)
      td_error = returns - q

      if self._clip_loss_grad:
          y = q.reshape(-1, 1)
          t = returns.reshape(-1, 1)
          loss = torch.nn.functional.huber_loss(y, t, reduction="mean") - self.eta * fg_prior.mean()
      else:
          loss = (td_error.pow(2) * weight).mean() - self.eta * fg_prior.mean()

      batch.weight = td_error  # prio-buffer
      loss.backward()
      self.optim.step()
      self._iter += 1
      return {"loss": loss.item()}

  def update(self,
    sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
    """Update the policy network and replay buffer.
    :param int sample_size: 0 means it will extract all the data from the buffer,
        otherwise it will sample a batch with given sample_size.
    :param ReplayBuffer buffer: the corresponding replay buffer.
    :return: A dict, including the data needed to be logged (e.g., loss) from
        ``policy.learn()``.
    """
    if buffer is None:
        return {}
    # Perform multiple updates
    self.updating = True
    for _ in range(self.update_num):
      batch, indices = buffer.sample(sample_size)
      batch = self.process_fn(batch, buffer, indices)
      result = self.learn(batch, **kwargs)
      self.post_process_fn(batch, buffer, indices)
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()
    self.updating = False
    return result