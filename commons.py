import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q)**2)) * torch.exp(-2. * logs_q)
  return kl


def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
  return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
  g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
  return g


def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(
      torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x


def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device
  
  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2,3) * mask
  return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm


class AdamWWarmup():
  def __init__(self, params, learning_rate, betas, eps, warmup_steps):
    self.dim_model = 256
    self.step_num = 1
    self.warmup_steps = warmup_steps

    self.params = params
    self.scheduler = "noam"
    self.warmup_steps = warmup_steps
    self.lr = learning_rate / (np.power(self.dim_model, -0.5) * np.power(warmup_steps, -0.5))
    self.betas = betas
    self.eps = eps

    self.step_num = 1
    self.cur_lr = self.lr * self._get_lr_scale()
    self._optim = torch.optim.AdamW(params, lr=self.cur_lr, betas=betas, eps=eps)
    self.param_groups = self._optim.param_groups


  def _get_lr_scale(self):
    return np.power(self.dim_model, -0.5) * np.min([np.power(self.step_num, -0.5), self.step_num * np.power(self.warmup_steps, -1.5)])


  def _update_learning_rate(self):
    self.step_num += 1

    self.cur_lr = self.lr * self._get_lr_scale()
    for param_group in self._optim.param_groups:
      param_group['lr'] = self.cur_lr
  def get_lr(self):
    return self.cur_lr

  def step(self):
    self._optim.step()
    self._update_learning_rate()

  def zero_grad(self):
    self._optim.zero_grad()

  def load_state_dict(self, d):
    self._optim.load_state_dict(d)

  def state_dict(self):
    return self._optim.state_dict()


# class WarmupSchedule(NoneSchedule):
#   def __init__(self, optimizer, lr, warmup_updates):
#     self.optimizer = optimizer
#     self.constant_lr = self.lr = lr
#     self.warmup_updates = warmup_updates
#     for param_group in optimizer.param_groups:
#       param_group['lr'] = self.lr
#     self.step(0)
#
#   def step(self, num_updates):
#     constant_lr = self.constant_lr
#     warmup = min(num_updates / self.warmup_updates, 1.0)
#     self.lr = max(constant_lr * warmup, 1e-7)
#     for param_group in self.optimizer.param_groups:
#       param_group['lr'] = self.lr
#     return self.lr
#
#   self.scheduler = WarmupSchedule(optimizer, hparams['lr'], hparams['warmup_updates'])
#   self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])
#
#   optimizer.zero_grad()
#   task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)