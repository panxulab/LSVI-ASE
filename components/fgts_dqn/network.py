import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional, Sequence


class FGTSLangevinDQNNet(nn.Module):
  """Reference: Human-level control through deep reinforcement learning.

  For advanced usage (how to customize the network), please refer to
  :ref:`build_the_network`.
  """

  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    device: Union[str, int, torch.device] = "cpu",
    features_only: bool = False,
    output_dim: Optional[int] = None,
    num_ensemble: int = 5,
    prior_scale: float = -1.,
  ) -> None:
    super().__init__()
    self.device = device
    self.num_ensemble = num_ensemble
    self.action_shape = action_shape
    # Set output dim
    if not features_only:
      self.output_dim = np.prod(action_shape)
    elif output_dim is not None:
      self.output_dim = output_dim
    self.feature_dim = None
    # Make ensemble nets
    net_list = []
    for i in range(self.num_ensemble):
      net = self.create_one_net(c, h, w, features_only)
      if prior_scale > 0:
        prior_net = self.create_one_net(c, h, w, features_only)
        net_list.append(NetworkWithPrior(net, prior_net, prior_scale))
      else:
        net_list.append(NetworkWithoutPrior(net))
    self.nets = nn.ModuleList(net_list)

  def create_one_net(
    self, c, h, w,
    features_only: bool = False,
  ) -> nn.Module:
    feature_net = nn.Sequential(
      nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
      nn.Flatten()
    )
    if self.feature_dim is None:
      with torch.no_grad():
        self.feature_dim = np.prod(feature_net(torch.zeros(1, c, h, w)).shape[1:])
    if not features_only:
      net = nn.Sequential(
        feature_net, nn.Linear(self.feature_dim, 512), nn.ReLU(inplace=True),
        nn.Linear(512, np.prod(self.action_shape))
      )
    else:
      net = nn.Sequential(
        feature_net, nn.Linear(self.feature_dim, self.output_dim),
        nn.ReLU(inplace=True)
      )
    return net.to(self.device)

  def forward(
    self,
    obs: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Dict[str, Any] = {},
    head: int = None
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: s -> Q(s, \*)."""
    obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
    logits, state = self.nets[head](obs)
    return logits, state


class NetworkWithPrior(nn.Module):
  """Combines network with additive untrainable "prior network"."""
  def __init__(self,
               network: nn.Module,
               prior_network: nn.Module,
               prior_scale: float = 1.):
    super().__init__()
    self._network = network
    self._prior_network = prior_network
    self._prior_scale = prior_scale

  def forward(
    self,
    obs: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Dict[str, Any] = {},
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: s -> Q(s, \*)."""
    q_values = self._network(obs)
    with torch.no_grad():
      prior_q_values = self._prior_network(obs)
    return q_values + self._prior_scale * prior_q_values.detach(), state


class NetworkWithoutPrior(nn.Module):
  """Combines network without additive untrainable "prior network"."""
  def __init__(self, network: nn.Module):
    super().__init__()
    self._network = network

  def forward(
    self,
    obs: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Dict[str, Any] = {},
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: s -> Q(s, \*)."""
    q_values = self._network(obs)
    return q_values, state