import h5py
import numpy as np
from numba import njit
from typing import Any, List, Optional, Tuple, Union

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import _alloc_by_keys_diff, _create_value


class FGTSReplayBuffer(ReplayBuffer):
  """:class:`~tianshou.data.ReplayBuffer` stores data generated from interaction \
  between the policy and environment.

  ReplayBuffer can be considered as a specialized form (or management) of Batch. It
  stores all the data in a batch with circular-queue style.

  For the example usage of ReplayBuffer, please check out Section Buffer in
  :doc:`/tutorials/concepts`.

  :param int size: the maximum size of replay buffer.
  :param int stack_num: the frame-stack sampling argument, should be greater than or
    equal to 1. Default to 1 (no stacking).
  :param bool ignore_obs_next: whether to store obs_next. Default to False.
  :param bool save_only_last_obs: only save the last obs/obs_next when it has a shape
    of (timestep, ...) because of temporal stacking. Default to False.
  :param bool sample_avail: the parameter indicating sampling only available index
    when using frame-stack sampling method. Default to False.
  """

  # Add first obs
  _reserved_keys = (
    "obs_first", "obs", "act", "rew", "terminated", "truncated", "done", "obs_next", "info",
    "policy"
  )
  _input_keys = (
    "obs_first", "obs", "act", "rew", "terminated", "truncated", "obs_next", "info", "policy"
  )

  # def __init__(
  #   self,
  #   size: int,
  #   stack_num: int = 1,
  #   ignore_obs_next: bool = False,
  #   save_only_last_obs: bool = False,
  #   sample_avail: bool = False,
  #   **kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
  # ) -> None:
  #   super().__init__(size, stack_num, ignore_obs_next, save_only_last_obs, sample_avail, **kwargs)

  @classmethod
  def from_data(
    cls, obs_first: h5py.Dataset, obs: h5py.Dataset, act: h5py.Dataset, rew: h5py.Dataset,
    terminated: h5py.Dataset, truncated: h5py.Dataset, done: h5py.Dataset,
    obs_next: h5py.Dataset
  ) -> "FGTSReplayBuffer":
    size = len(obs)
    assert all(len(dset) == size for dset in [obs_first, obs, act, rew, terminated,
                          truncated, done, obs_next]), \
      "Lengths of all hdf5 datasets need to be equal."
    buf = cls(size)
    if size == 0:
      return buf
    batch = Batch(
      obs_first=obs_first, # Add first obs
      obs=obs,
      act=act,
      rew=rew,
      terminated=terminated,
      truncated=truncated,
      done=done,
      obs_next=obs_next
    )
    buf.set_batch(batch)
    buf._size = size
    return buf

  def add(
    self,
    batch: Batch,
    buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add a batch of data into replay buffer.

    :param Batch batch: the input data batch. Its keys must belong to the 7
      input keys, and "obs", "act", "rew", "terminated", "truncated" is
      required.
    :param buffer_ids: to make consistent with other buffer's add function; if it
      is not None, we assume the input batch's first dimension is always 1.

    Return (current_index, episode_reward, episode_length, episode_start_index). If
    the episode is not finished, the return value of episode_length and
    episode_reward is 0.
    """
    # preprocess batch
    new_batch = Batch()
    for key in set(self._input_keys).intersection(batch.keys()):
      new_batch.__dict__[key] = batch[key]
    batch = new_batch
    batch.__dict__["done"] = np.logical_or(batch.terminated, batch.truncated)
    # Add first obs
    assert set(["obs_first", "obs", "act", "rew", "terminated", "truncated",
          "done"]).issubset(batch.keys())
    stacked_batch = buffer_ids is not None
    if stacked_batch:
      assert len(batch) == 1
    # Add first obs
    if self._save_only_last_obs:
      batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
      batch.obs_first = batch.obs_first[:, -1] if stacked_batch else batch.obs_first[-1]
    if not self._save_obs_next:
      batch.pop("obs_next", None)
    elif self._save_only_last_obs:
      batch.obs_next = (
        batch.obs_next[:, -1] if stacked_batch else batch.obs_next[-1]
      )
    # get ptr
    if stacked_batch:
      rew, done = batch.rew[0], batch.done[0]
    else:
      rew, done = batch.rew, batch.done
    ptr, ep_rew, ep_len, ep_idx = list(
      map(lambda x: np.array([x]), self._add_index(rew, done))
    )
    try:
      self._meta[ptr] = batch
    except ValueError:
      stack = not stacked_batch
      batch.rew = batch.rew.astype(float)
      batch.done = batch.done.astype(bool)
      batch.terminated = batch.terminated.astype(bool)
      batch.truncated = batch.truncated.astype(bool)
      if self._meta.is_empty():
        self._meta = _create_value(  # type: ignore
          batch, self.maxsize, stack)
      else:  # dynamic key pops up in batch
        _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
      self._meta[ptr] = batch
    return ptr, ep_rew, ep_len, ep_idx

  def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
    """Return a data batch: self[index].

    If stack_num is larger than 1, return the stacked obs and obs_next with shape
    (batch, len, ...).
    """
    if isinstance(index, slice):  # change slice to np array
      # buffer[:] will get all available data
      indices = self.sample_indices(0) if index == slice(None) \
        else self._indices[:len(self)][index]
    else:
      indices = index  # type: ignore
    # raise KeyError first instead of AttributeError,
    # to support np.array([FGTSReplayBuffer()])
    obs = self.get(indices, "obs")
    # Get first obs
    obs_first = self.get(indices, "obs_first")
    if self._save_obs_next:
      obs_next = self.get(indices, "obs_next", Batch())
    else:
      obs_next = self.get(self.next(indices), "obs", Batch())
    return Batch(
      obs_first=obs_first, # Add first obs
      obs=obs,
      act=self.act[indices],
      rew=self.rew[indices],
      terminated=self.terminated[indices],
      truncated=self.truncated[indices],
      done=self.done[indices],
      obs_next=obs_next,
      info=self.get(indices, "info", Batch()),
      policy=self.get(indices, "policy", Batch()),
    )


class FGTSReplayBufferManager(FGTSReplayBuffer):
  """ReplayBufferManager contains a list of ReplayBuffer with exactly the same \
  configuration.

  These replay buffers have contiguous memory layout, and the storage space each
  buffer has is a shallow copy of the topmost memory.

  :param buffer_list: a list of ReplayBuffer needed to be handled.

  .. seealso::

    Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
  """

  def __init__(self, buffer_list: List[FGTSReplayBuffer]) -> None:
    self.buffer_num = len(buffer_list)
    self.buffers = np.array(buffer_list, dtype=object)
    offset, size = [], 0
    buffer_type = type(self.buffers[0])
    kwargs = self.buffers[0].options
    for buf in self.buffers:
      assert buf._meta.is_empty()
      assert isinstance(buf, buffer_type) and buf.options == kwargs
      offset.append(size)
      size += buf.maxsize
    self._offset = np.array(offset)
    self._extend_offset = np.array(offset + [size])
    self._lengths = np.zeros_like(offset)
    super().__init__(size=size, **kwargs)
    self._compile()
    self._meta: Batch

  def _compile(self) -> None:
    lens = last = index = np.array([0])
    offset = np.array([0, 1])
    done = np.array([False, False])
    _prev_index(index, offset, done, last, lens)
    _next_index(index, offset, done, last, lens)

  def __len__(self) -> int:
    return int(self._lengths.sum())

  def reset(self, keep_statistics: bool = False) -> None:
    self.last_index = self._offset.copy()
    self._lengths = np.zeros_like(self._offset)
    for buf in self.buffers:
      buf.reset(keep_statistics=keep_statistics)

  def _set_batch_for_children(self) -> None:
    for offset, buf in zip(self._offset, self.buffers):
      buf.set_batch(self._meta[offset:offset + buf.maxsize])

  def set_batch(self, batch: Batch) -> None:
    super().set_batch(batch)
    self._set_batch_for_children()

  def unfinished_index(self) -> np.ndarray:
    return np.concatenate(
      [
        buf.unfinished_index() + offset
        for offset, buf in zip(self._offset, self.buffers)
      ]
    )

  def prev(self, index: Union[int, np.ndarray]) -> np.ndarray:
    if isinstance(index, (list, np.ndarray)):
      return _prev_index(
        np.asarray(index), self._extend_offset, self.done, self.last_index,
        self._lengths
      )
    else:
      return _prev_index(
        np.array([index]), self._extend_offset, self.done, self.last_index,
        self._lengths
      )[0]

  def next(self, index: Union[int, np.ndarray]) -> np.ndarray:
    if isinstance(index, (list, np.ndarray)):
      return _next_index(
        np.asarray(index), self._extend_offset, self.done, self.last_index,
        self._lengths
      )
    else:
      return _next_index(
        np.array([index]), self._extend_offset, self.done, self.last_index,
        self._lengths
      )[0]

  def update(self, buffer: FGTSReplayBuffer) -> np.ndarray:
    """The ReplayBufferManager cannot be updated by any buffer."""
    raise NotImplementedError
  
  def add(
    self,
    batch: Batch,
    buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add a batch of data into ReplayBufferManager.

    Each of the data's length (first dimension) must equal to the length of
    buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

    Return (current_index, episode_reward, episode_length, episode_start_index). If
    the episode is not finished, the return value of episode_length and
    episode_reward is 0.
    """
    # preprocess batch
    new_batch = Batch()
    for key in set(self._reserved_keys).intersection(batch.keys()):
      new_batch.__dict__[key] = batch[key]
    batch = new_batch
    batch.__dict__["done"] = np.logical_or(batch.terminated, batch.truncated)
    # Add first obs
    assert set(["obs_first", "obs", "act", "rew", "terminated", "truncated",
          "done"]).issubset(batch.keys())
    if self._save_only_last_obs:
      batch.obs = batch.obs[:, -1]
      batch.obs_first = batch.obs_first[:, -1]
    if not self._save_obs_next:
      batch.pop("obs_next", None)
    elif self._save_only_last_obs:
      batch.obs_next = batch.obs_next[:, -1]
    # get index
    if buffer_ids is None:
      buffer_ids = np.arange(self.buffer_num)
    ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
    for batch_idx, buffer_id in enumerate(buffer_ids):
      ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(
        batch.rew[batch_idx], batch.done[batch_idx]
      )
      ptrs.append(ptr + self._offset[buffer_id])
      ep_lens.append(ep_len)
      ep_rews.append(ep_rew)
      ep_idxs.append(ep_idx + self._offset[buffer_id])
      self.last_index[buffer_id] = ptr + self._offset[buffer_id]
      self._lengths[buffer_id] = len(self.buffers[buffer_id])
    ptrs = np.array(ptrs)
    try:
      self._meta[ptrs] = batch
    except ValueError:
      batch.rew = batch.rew.astype(float)
      batch.done = batch.done.astype(bool)
      batch.terminated = batch.terminated.astype(bool)
      batch.truncated = batch.truncated.astype(bool)
      if self._meta.is_empty():
        self._meta = _create_value(  # type: ignore
          batch, self.maxsize, stack=False)
      else:  # dynamic key pops up in batch
        _alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
      self._set_batch_for_children()
      self._meta[ptrs] = batch
    return ptrs, np.array(ep_rews), np.array(ep_lens), np.array(ep_idxs)

  def sample_indices(self, batch_size: int) -> np.ndarray:
    if batch_size < 0:
      return np.array([], int)
    if self._sample_avail and self.stack_num > 1:
      all_indices = np.concatenate(
        [
          buf.sample_indices(0) + offset
          for offset, buf in zip(self._offset, self.buffers)
        ]
      )
      if batch_size == 0:
        return all_indices
      else:
        return np.random.choice(all_indices, batch_size)
    if batch_size == 0:  # get all available indices
      sample_num = np.zeros(self.buffer_num, int)
    else:
      buffer_idx = np.random.choice(
        self.buffer_num, batch_size, p=self._lengths / self._lengths.sum()
      )
      sample_num = np.bincount(buffer_idx, minlength=self.buffer_num)
      # avoid batch_size > 0 and sample_num == 0 -> get child's all data
      sample_num[sample_num == 0] = -1

    return np.concatenate(
      [
        buf.sample_indices(bsz) + offset
        for offset, buf, bsz in zip(self._offset, self.buffers, sample_num)
      ]
    )


@njit
def _prev_index(
  index: np.ndarray,
  offset: np.ndarray,
  done: np.ndarray,
  last_index: np.ndarray,
  lengths: np.ndarray,
) -> np.ndarray:
  index = index % offset[-1]
  prev_index = np.zeros_like(index)
  for start, end, cur_len, last in zip(offset[:-1], offset[1:], lengths, last_index):
    mask = (start <= index) & (index < end)
    cur_len = max(1, cur_len)
    if np.sum(mask) > 0:
      subind = index[mask]
      subind = (subind - start - 1) % cur_len
      end_flag = done[subind + start] | (subind + start == last)
      prev_index[mask] = (subind + end_flag) % cur_len + start
  return prev_index


@njit
def _next_index(
  index: np.ndarray,
  offset: np.ndarray,
  done: np.ndarray,
  last_index: np.ndarray,
  lengths: np.ndarray,
) -> np.ndarray:
  index = index % offset[-1]
  next_index = np.zeros_like(index)
  for start, end, cur_len, last in zip(offset[:-1], offset[1:], lengths, last_index):
    mask = (start <= index) & (index < end)
    cur_len = max(1, cur_len)
    if np.sum(mask) > 0:
      subind = index[mask]
      end_flag = done[subind] | (subind == last)
      next_index[mask] = (subind - start + 1 - end_flag) % cur_len + start
  return next_index


class FGTSVectorReplayBuffer(FGTSReplayBufferManager):
  """VectorReplayBuffer contains n ReplayBuffer with the same size.

  It is used for storing transition from different environments yet keeping the order
  of time.

  :param int total_size: the total size of VectorReplayBuffer.
  :param int buffer_num: the number of ReplayBuffer it uses, which are under the same
    configuration.

  Other input arguments (stack_num/ignore_obs_next/save_only_last_obs/sample_avail)
  are the same as :class:`~tianshou.data.ReplayBuffer`.

  .. seealso::

    Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
  """

  def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
    assert buffer_num > 0
    size = int(np.ceil(total_size / buffer_num))
    buffer_list = [FGTSReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
    super().__init__(buffer_list)