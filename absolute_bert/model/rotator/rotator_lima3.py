# 最主要的差別就是 history 也會被 rotate

import torch
from torch import nn
import itertools

class Multihead_Lima(nn.Module):
  def __init__(self, dim=64, num_heads=4, fix_max_value=1):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
      
    if fix_max_value: 
        assert fix_max_value > 0
        self.M = torch.nn.Parameter(torch.tensor(fix_max_value), requires_grad=False)
    else:
        raise NotImplementedError
        
    self.proportion = torch.nn.Parameter(torch.rand(self.num_heads, self.dim)) # weight 的 range 會限制在 M 下面的一個比例: 1/(e^{proportion}+1)
    self.lima_shape = torch.nn.Parameter(torch.rand(self.num_heads, self.dim)) # 類似之前的 b/a, 會用 sigmoid 把 range 限制在 -0.25 ~ 0.25
    self.Theta = torch.nn.Parameter(torch.rand(self.num_heads, self.dim, self.dim)*torch.pi*2 - torch.pi, requires_grad=True)
    self.layer_norm = nn.LayerNorm(self.dim)

  def forward(self, theta):
    """theta: shape: [num_heads, batch_size, embedding_dim]"""
    b_a = ((torch.sigmoid(self.lima_shape) - 0.5) / 2)[:, None, :]  # b/a
    scale_controlling_coefficient = self.M / (2*(torch.pow(torch.e, self.proportion)+1))
    
    o = self.M + scale_controlling_coefficient[:, None, :] * (-1 - b_a + theta.cos() + b_a*(2*theta).cos())
    # print(f'multihead_lima: {o.shape=}, {self.Theta.shape=}')
    raw_resulting_angle = torch.einsum('hbm,hmn->hbn', o, self.Theta) #shape: [heads, batch_size, dim]
    return self.layer_norm(raw_resulting_angle)

class Output(nn.Module):
  def __init__(self, vocab_size, dim=64, rotary_denom=.5):
    super().__init__()
    self.dim = dim
    self.vocab_size = vocab_size
    self.dimension_nums = torch.nn.Parameter(torch.arange(0, dim, dtype=torch.float), requires_grad=False)
    self.initial_rotary_denom = rotary_denom  # 參考 transformer 的 position embedding 分母的 10000
    self.rotary_denom = torch.nn.Parameter(torch.tensor(self.initial_rotary_denom, dtype=torch.float))
    self.embedding_real = torch.nn.Embedding(vocab_size, self.dim)
    self.embedding_imag = torch.nn.Embedding(vocab_size, self.dim)

  def embedding(self, input_ids):
    return torch.complex(self.embedding_real(input_ids), self.embedding_imag(input_ids))
    # return torch.stack([self.embedding_real(input_ids), self.embedding_imag(input_ids)], dim=-1)

  def all_word_embeddings(self):
      return torch.complex(self.embedding_real.weight, self.embedding_imag.weight).clone()

  def forward(self, histories, sources, t, word_angles):
    time_angle = 1/self.rotary_denom**(self.dimension_nums/self.dim)
    total_angles = t*(time_angle+word_angles) + histories

    # cos, sin = total_angles.cos(), total_angles.sin()
    # embedding = self.embedding(sources)
    # real, imag = embedding[..., 0], embedding[..., 1]
    # return torch.stack([real*cos-imag*sin, real*sin+imag*cos], dim=-1)

    rotation = torch.complex(total_angles.cos(), total_angles.sin())
    return self.embedding(sources)*rotation

class Rotator(nn.Module):
  def __init__(self, vocab_size, dim=64, num_heads=4, depth=3, rotary_denom=.5, **kwargs):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.num_heads = num_heads
    self.depth = depth
    self.limas = nn.ModuleList([Multihead_Lima(dim=self.dim, num_heads=num_heads) for _ in range(self.depth)])
    self.initial_history = nn.ParameterList(
        [torch.nn.Parameter(torch.randn(self.num_heads, 1, self.dim), requires_grad=True) for _ in range(self.depth)]
    )

    self.angle_embedding = torch.nn.Embedding(vocab_size, self.dim)
    self.predictor = Output(vocab_size, dim=dim, rotary_denom=rotary_denom)
    
    self.head_clock_denoms = torch.nn.Parameter(torch.tensor([.5]*self.num_heads))

  def embedding(self, input_ids):
    return self.predictor.embedding(input_ids)

  def forward(self, data, batch_sizes, **kwargs):

    histories = [history.clone() for history in self.initial_history]
    head_clocks = (1/self.head_clock_denoms[:, None]**(self.predictor.dimension_nums/self.dim))[:, None, :]

    for t, (batch_accum, batch_size, next_batch_size) in enumerate(zip(itertools.accumulate(batch_sizes, initial=0), batch_sizes, batch_sizes[1:])):
      batch = data[batch_accum:batch_accum+batch_size][:next_batch_size]
      theta = self.angle_embedding(batch)
      # print(f"{t=}: {theta.shape=}, {[history.shape for history in self.initial_history]=}")

      accum = 0
      
      for lima_num, lima in enumerate(self.limas):
        # print(f'{(theta-accum).shape=}, {histories[lima_num][:, :theta.shape[0], :].shape=}')
        
        diff = theta-accum-histories[lima_num][:, :theta.shape[0], :]
        # print(f'Rotator_with_time.forward, {diff.shape=}')
        history = lima(diff)
        # print(f"In Rotator, {batch_accum=}, {lima_num=}, {diff=}")
        histories[lima_num] = history - head_clocks
        accum += history.sum(dim=0)

      predicts = self.predictor(accum, batch, t, theta)
      labels = data[batch_accum+batch_size:batch_accum+batch_size+next_batch_size]

      yield predicts, self.embedding(labels)
    
  def pairwise_distance(self, A: torch.complex, B: torch.complex, distance='cos', eps=1e-8):
    """
    A: MxD, B: NxD
    distance: 'cos', 'l2'
    """
    if distance == 'cos':
      product = torch.einsum('md,nd->mn', A, B.conj())
      norms = A.norm(dim=1)[...,None] * B.norm(dim=1)[None,...] + eps
      return (product / norms).real
        
    if distance == 'l2':
      AT = A[..., None]
      BT = B.T[None, ...]
      # return (AT.real**2+AT.imag**2).sum(dim=0) + (BT.real**2+BT.imag**2).sum(dim=-1) - torch.einsum('DM...,...ND->MN', AT, BT.conj())
      return (AT - BT).norm(dim=1)
    
  def inference(self, input_ids, histories, t, return_updated_history=False, distance='l2'):
    
    head_clocks = (1/self.head_clock_denoms[:, None]**(self.predictor.dimension_nums/self.dim))[:, None, :]
    batch = self.angle_embedding(input_ids)
    accum = 0  
    for lima_num, lima in enumerate(self.limas):
        # print(f'inference: {(batch-accum).shape}, {histories[lima_num].shape}')
        history = lima(batch - accum - histories[lima_num])
        histories[lima_num] = history - head_clocks
        accum += history.sum(dim=0)
        
    predicts = self.predictor.forward(accum, input_ids, t, batch)
    pdist = self.pairwise_distance(predicts, self.predictor.all_word_embeddings(), distance=distance)
    minargs = pdist.argsort(dim=1)

    if return_updated_history:
        return minargs, histories
        
    return minargs