import math

import numpy as np
import torch
from torch import Tensor
from typing import List, Optional

from torch.optim.sgd import *
from torch.optim.adam import *
from torch.optim.rmsprop import *
from torch.optim.optimizer import *

from components.langevin_dqn.optim import *


class aSGLD(Adam):
  """
  Implementation of Adam SGLD based on: http://arxiv.org/abs/2009.09535
  Built on PyTorch Adam implementation.
  Note that there is no bias correction in the original description of Adam SGLD.
  """

  def __init__(
          self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
          weight_decay=0, amsgrad=False,
          noise_scale=0.01, a=1.0
  ):
    defaults = dict(
      lr=lr, betas=betas, eps=eps,
      weight_decay=weight_decay, amsgrad=amsgrad
    )
    super(aSGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.a = a

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])

          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])

      adam_sgld(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=group['amsgrad'],
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'],
        noise_scale=self.noise_scale,
        a=self.a
      )
    return loss


def adam_sgld(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        noise_scale: float,
        a: float
):
  """Functional API that performs Adam SGLD algorithm computation.
  See :class:`~torch.optim.Adam` for details.
  """
  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = max_exp_avg_sqs[i].sqrt().add_(eps)
    else:
      denom = exp_avg_sq.sqrt().add_(eps)

    # Add pure gradient
    param.add_(grad, alpha=-lr)
    # Add the adaptive bias term
    am = a * exp_avg
    param.addcdiv_(am, denom, value=-lr)
    # Add noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    param.add_(noise_scale * math.sqrt(2.0 * lr) * grad_perturb)



class EulerULMC(SGD):
  """
  Built on PyTorch SGD implementation
  Change the update order from EulerULMC
  """
  def __init__(
          self, params, lr=1e-2, gamma=0.1, weight_decay=0,
          noise_scale=0.01
  ):
    defaults = dict(
      lr=lr, weight_decay=weight_decay, nesterov=False
    )
    super(EulerULMC, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.gamma = gamma

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      d_p_list = []
      momentum_buffer_list = []
      has_sparse_grad = False

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          d_p_list.append(p.grad)
          if p.grad.is_sparse:
            has_sparse_grad = True

          state = self.state[p]
          if 'momentum_buffer' not in state:
            momentum_buffer_list.append(None)
          else:
            momentum_buffer_list.append(state['momentum_buffer'])
      eulerulmc(
        params_with_grad,
        d_p_list,
        momentum_buffer_list,
        weight_decay=group['weight_decay'],
        lr=group['lr'],
        has_sparse_grad=has_sparse_grad,
        noise_scale=self.noise_scale,
        gamma=self.gamma
      )

      # Update momentum_buffers in state
      for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
        state = self.state[p]
        state['momentum_buffer'] = momentum_buffer
    return loss


def eulerulmc(
        params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        lr: float,
        has_sparse_grad: bool,
        noise_scale: float,
        gamma: float
):
  """
  See :class:`~torch.optim.SGD` for details.
  """
  for i, param in enumerate(params):
    d_p = d_p_list[i]
    if weight_decay != 0:
      d_p = d_p.add(param, alpha=weight_decay)
    # Compute momentum
    buf = momentum_buffer_list[i]
    if buf is None:
      buf = torch.zeros_like(d_p)
      momentum_buffer_list[i] = buf
    # update buf
    buf.mul_(1-lr*gamma).add_(d_p, alpha=lr)
    # Add buf noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    buf.add_(noise_scale * math.sqrt(2.0 * lr * gamma) * grad_perturb)
    # update params
    param.add_(buf, alpha=-lr)


# Move the noise scale out from the second order P.
class aULMC(Adam):
  """
  Implementation of Adam ULMC
  Built on PyTorch Adam implementation.
  Note that there is no bias correction in the original description of Adam ULMC.
  """

  def __init__(
          self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
          weight_decay=0, amsgrad=False,
          noise_scale=0.01, gamma=1, a=1.0
  ):
    defaults = dict(
      lr=lr, betas=betas, eps=eps,
      weight_decay=weight_decay, amsgrad=amsgrad
    )
    super(aULMC, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.a = a
    self.gamma = gamma

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      momentum_buffer_list = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Momentum Buffer
            state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])
          momentum_buffer_list.append(state['momentum_buffer'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])

          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])
      adam_ulmc(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        momentum_buffer_list,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=group['amsgrad'],
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'],
        noise_scale=self.noise_scale,
        a=self.a,
        gamma=self.gamma
      )
    return loss


def adam_ulmc(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        momentum_buffer_list: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        noise_scale: float,
        a: float,
        gamma: float
):
  """Functional API that performs Adam SGLD algorithm computation.
  See :class:`~torch.optim.Adam` for details.
  """
  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]
    buf = momentum_buffer_list[i]

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = max_exp_avg_sqs[i].sqrt().add_(eps)
    else:
      denom = exp_avg_sq.sqrt().add_(eps)
    buf.mul_(1 - lr * gamma).add_(grad, alpha=lr)
    am = a * exp_avg
    buf.addcdiv_(am, denom, value=lr)
    # Add buf noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    # Update params
    param.add_(buf, alpha=-lr)
    param.add_(noise_scale * math.sqrt(2.0 * lr * gamma) * grad_perturb)




class pSGLD(RMSprop):
  """
  Implementation of pSGLD based on https://arxiv.org/abs/1512.07666
  Built on PyTorch RMSprop implementation
  """
  def __init__(
    self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
    noise_scale=1.0
  ):
    defaults = dict(
      lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
      momentum=momentum, centered=centered
    )
    super(pSGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      square_avgs = []
      grad_avgs = []
      momentum_buffer_list = []

      for p in group['params']:
        if p.grad is None:
          continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError('RMSprop does not support sparse gradients')
        
        grads.append(p.grad)
        state = self.state[p]
        
        # State initialization
        if len(state) == 0:
          state['step'] = 0
          state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
          if group['momentum'] > 0:
            state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
          if group['centered']:
            state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        
        square_avgs.append(state['square_avg'])
        
        if group['momentum'] > 0:
          momentum_buffer_list.append(state['momentum_buffer'])
        if group['centered']:
          grad_avgs.append(state['grad_avg'])
        state['step'] += 1

      p_sgld(
        params_with_grad,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
        lr=group['lr'],
        alpha=group['alpha'],
        eps=group['eps'],
        weight_decay=group['weight_decay'],
        momentum=group['momentum'],
        centered=group['centered'],
        noise_scale=self.noise_scale
      )
    return loss

def p_sgld(
  params: List[Tensor],
  grads: List[Tensor],
  square_avgs: List[Tensor],
  grad_avgs: List[Tensor],
  momentum_buffer_list: List[Tensor],
  *,
  lr: float,
  alpha: float,
  eps: float,
  weight_decay: float,
  momentum: float,
  centered: bool,
  noise_scale: float
):
  r"""Functional API that performs rmsprop algorithm computation.
  See :class:`~torch.optim.RMSProp` for details.
  """
  for i, param in enumerate(params):
    grad = grads[i]
    square_avg = square_avgs[i]

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    square_avg.mul_(alpha).addcmul_(grad, grad, value=1-alpha)

    if centered:
      grad_avg = grad_avgs[i]
      grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
      avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
    else:
      avg = square_avg.sqrt().add_(eps)

    if momentum > 0:
      buf = momentum_buffer_list[i]
      buf.mul_(momentum).addcdiv_(grad, avg)
      param.add_(buf, alpha=-lr)
    else:
      param.addcdiv_(grad, avg, value=-lr)
    
    # Add noise to gradient as grad_perturb
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device) / avg.sqrt()
    param.add_(noise_scale * math.sqrt(2*lr) * grad_perturb)





class mSGLD(SGD):
  """
  Implementation of Momentum SGLD based on: http://arxiv.org/abs/2009.09535
  Built on PyTorch SGD implementation
  """
  def __init__(
    self, params, lr=1e-2, beta=0.99, weight_decay=0,
    noise_scale=0.01, a=1.0
  ):
    defaults = dict(
      lr=lr, momentum=beta, dampening=1-beta,
      weight_decay=weight_decay, nesterov=False
    )
    super(mSGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.a = a

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      d_p_list = []
      momentum_buffer_list = []
      has_sparse_grad = False

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          d_p_list.append(p.grad)
          if p.grad.is_sparse:
            has_sparse_grad = True

          state = self.state[p]
          if 'momentum_buffer' not in state:
            momentum_buffer_list.append(None)
          else:
            momentum_buffer_list.append(state['momentum_buffer'])
      
      momentum_sgd(
        params_with_grad,
        d_p_list,
        momentum_buffer_list,
        weight_decay=group['weight_decay'],
        momentum=group['momentum'],
        lr=group['lr'],
        dampening=group['dampening'],
        maximize=group['maximize'],
        has_sparse_grad=has_sparse_grad,
        noise_scale=self.noise_scale,
        a=self.a
      )
      # Update momentum_buffers in state
      for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
        state = self.state[p]
        state['momentum_buffer'] = momentum_buffer

    return loss

def momentum_sgd(
  params: List[Tensor],
  d_p_list: List[Tensor],
  momentum_buffer_list: List[Optional[Tensor]],
  *,
  weight_decay: float,
  momentum: float,
  lr: float,
  dampening: float,
  maximize: bool,
  has_sparse_grad: bool,
  noise_scale: float,
  a: float
):
  """Functional API that performs Momentum SGD algorithm computation.
  See :class:`~torch.optim.SGD` for details.
  """
  for i, param in enumerate(params):
    d_p = d_p_list[i]
    if weight_decay != 0:
      d_p = d_p.add(param, alpha=weight_decay)
    # Compute momentum
    if momentum != 0:
      buf = momentum_buffer_list[i]
      if buf is None:
        buf = torch.clone(d_p).detach()
        momentum_buffer_list[i] = buf
      else:
        buf.mul_(momentum).add_(d_p, alpha=1-dampening)
    # Add the adaptive bias term
    d_p = d_p.add(buf, alpha=a)
    alpha = lr if maximize else -lr
    param.add_(d_p, alpha=alpha)
    # Add noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    param.add_(noise_scale * math.sqrt(2.0*lr) * grad_perturb)


class SGLD(SGD):
  r"""Implementation of Stochastic Gradient Langevin Dynamics (SGLD): http://www.icml-2011.org/papers/398_icmlpaper.pdf
  Built on PyTorch SGD implementation.
  """
  def __init__(self, params, lr=1e-2, weight_decay=0, noise_scale=0.01):
    defaults = dict(lr=lr, weight_decay=weight_decay)
    super(SGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      d_p_list = []
      
      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          d_p_list.append(p.grad)
          state = self.state[p]
      
      sgld(
        params_with_grad,
        d_p_list,
        weight_decay=group['weight_decay'],
        lr=group['lr'],
        maximize=group['maximize'],
        noise_scale=self.noise_scale,
      )
    return loss

def sgld(params: List[Tensor],
    d_p_list: List[Tensor],
    *,
    weight_decay: float,
    lr: float,
    maximize: bool,
    noise_scale: float
  ):
  """Functional API that performs SLGD algorithm computation.
  See :class:`~torch.optim.SGD` for details.
  """
  for i, param in enumerate(params):
    d_p = d_p_list[i]
    if weight_decay != 0:
      d_p = d_p.add(param, alpha=weight_decay)
    alpha = lr if maximize else -lr
    # SGD update
    param.add_(d_p, alpha=alpha)
    # Add noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    param.add_(noise_scale * math.sqrt(2.0*lr) * grad_perturb)


class AntiaSGLD(Adam):
  """
  Adam SGLD with anticorrelated noise
  """
  def __init__(
    self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0, amsgrad=False,
    noise_scale=0.01, a=1.0
  ):
    defaults = dict(
      lr=lr, betas=betas, eps=eps, 
      weight_decay=weight_decay, amsgrad=amsgrad
    )
    super(AntiaSGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.a = a

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      old_noises = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Old noise used to compute anticorrelated noise
            state['old_noise'] = torch.zeros_like(p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])
          
          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])
          # record previous noise
          old_noises.append(state['old_noise'])

      new_noises = anti_adam_sgld(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        old_noises,
        amsgrad=group['amsgrad'],
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'],
        noise_scale=self.noise_scale,
        a=self.a
      )

      # Update state noise
      for i, p in enumerate(group['params']):
        if p.grad is not None:
          self.state[p]['old_noise'] = new_noises[i]

    return loss

def anti_adam_sgld(
  params: List[Tensor],
  grads: List[Tensor],
  exp_avgs: List[Tensor],
  exp_avg_sqs: List[Tensor],
  max_exp_avg_sqs: List[Tensor],
  state_steps: List[int],
  old_noises: List[Tensor],
  *,
  amsgrad: bool,
  beta1: float,
  beta2: float,
  lr: float,
  weight_decay: float,
  eps: float,
  noise_scale: float,
  a: float
):
  """Functional API that performs Adam SGLD algorithm computation with anticorrelated noise.
  See :class:`~torch.optim.Adam` for details.
  """
  new_noises = []
  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]
    old_noise = old_noises[i]

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = max_exp_avg_sqs[i].sqrt().add_(eps)
    else:
      denom = exp_avg_sq.sqrt().add_(eps)
    
    # Add pure gradient
    param.add_(grad, alpha=-lr)
    # Add the adaptive bias term
    am = a * exp_avg
    param.addcdiv_(am, denom, value=-lr)
    # Add noise
    new_noise = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    grad_perturb = new_noise - old_noise
    param.add_(noise_scale * math.sqrt(2.0*lr) * grad_perturb)
    new_noises.append(new_noise)
  return new_noises
