import numpy as np
import torch
from torch import nn
import itertools

class Multihead_GRUcell(nn.Module):
  def __init__(self, dim=64, hidden_dim=32, num_heads=4):
    super().__init__()
    
    self.dim = dim
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads

    # reset gate, proportion gate
    self.fc_sigmoid = nn.Linear(self.num_heads*(self.hidden_dim+self.dim), 2*self.num_heads)
    self.sigmoid_w = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim+self.dim, 2))
    self.sigmoid_b = nn.Parameter(torch.randn(self.num_heads, 2))
    self.sigmoid = nn.Sigmoid()
    self.fc_tanh = nn.Linear(self.num_heads*(self.hidden_dim+self.dim),
                              self.num_heads*self.hidden_dim)
    self.tanh_w = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim+self.dim, self.hidden_dim))
    self.tanh_b = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim))
    self.tanh = nn.Tanh()
    
    self.fc_out = torch.nn.Linear(self.num_heads*self.hidden_dim, self.num_heads*self.dim)
    self.out_w = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim, self.dim))
    self.out_b = nn.Parameter(torch.randn(self.num_heads, self.dim))
    # self.layer_norm = nn.LayerNorm(self.history_dim_out, bias=True)

  def forward(self, x, h):
    """
    x: shape [batch_size, dim]
    h: shape [batch_size, num_heads, hidden_dim]
    """

    x_ = x[:, None, :].expand(-1, self.num_heads, -1)
    sigmoid_input = torch.concat([h, x_], dim=2)
    sigmoids = self.sigmoid(torch.einsum('bhm,hmn->bhn', sigmoid_input, self.sigmoid_w) + self.sigmoid_b)
    
    tanh_input = torch.concat([sigmoids[:, :, :1]*h, x_], dim=2)
    tanh = self.tanh(torch.einsum('bhm,hmn->bhn', tanh_input, self.tanh_w) + self.tanh_b)

    proportion = sigmoids[:, :, 1:]
    new_h = (1-proportion)*h + proportion*tanh
    
    # output = self.fc_out(new_h).view(batch_size, self.num_heads, self.dim)
    output = torch.einsum('bhm,hmn->bhn', new_h, self.out_w) + self.out_b
    
    return output, new_h

class Output(nn.Module):
  def __init__(self, vocab_size, dim=64, rotary_denom=.5, local_coordinating=False, add_time_angles=False, **kwargs):
    super().__init__()
    self.dim = dim
    self.vocab_size = vocab_size
    self.dimension_indices = torch.nn.Parameter(torch.arange(0, dim, dtype=torch.float), requires_grad=False)
    self.initial_rotary_denom = rotary_denom  # 參考 transformer 的 position embedding 分母的 10000
    self.log_rotary_denom = torch.nn.Parameter(torch.tensor(self.initial_rotary_denom, dtype=torch.float))
    self.embedding_real = torch.nn.Embedding(vocab_size, self.dim)
    self.embedding_imag = torch.nn.Embedding(vocab_size, self.dim)
    
    self.base_vec = nn.Parameter(torch.randn([1,dim])/np.sqrt(dim))
    self.tanh = nn.Tanh()
    
    self.local_coordinating = local_coordinating
    self.add_time_angles = add_time_angles

  def embedding(self, input_ids):
    return torch.complex(self.embedding_real(input_ids), self.embedding_imag(input_ids))
    # return torch.stack([self.embedding_real(input_ids), self.embedding_imag(input_ids)], dim=-1)

  def all_word_embeddings(self):
    return torch.complex(self.embedding_real.weight, self.embedding_imag.weight).clone()

  def forward(self, angles, word_angles=None):
    total_angles = angles
    if self.add_time_angles:
      total_angles += 1 / (torch.e**self.log_rotary_denom)**(self.dimension_indices/self.dim) # shape: [1,dim]
    if self.local_coordinating:
      total_angles += word_angles

    total_angles = self.tanh(total_angles) * 3
    rotation = torch.complex(total_angles.cos(), total_angles.sin())
    return self.base_vec*rotation

class Rotator(nn.Module):
  def __init__(self, vocab_size, dim=64, hidden_dim=32, history_dim=None, hippo_dim=16, num_hippo_heads=4, num_heads=4, depth=3, rotary_denom=.5, scale_rank=4, **kwargs):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.hidden_dim = hidden_dim
    self.history_dim = hidden_dim if (history_dim==None) else history_dim 
    self.num_heads = num_heads
    self.depth = depth
    self.scale_rank = scale_rank  # 預測下一個 vector 的 norm
    
    self.limas = nn.ModuleList([Multihead_GRUcell(
      dim = self.dim,
      hidden_dim = self.hidden_dim,
      num_heads = self.num_heads
    ) for _ in range(self.depth)])
    
    self.predictor = Output(vocab_size, dim=self.dim, rotary_denom=rotary_denom, **kwargs) # for rotate
    self.scale_fc = nn.Linear(self.history_dim*self.depth, self.scale_rank)
    self.scale_act = nn.GELU()
    self.angle_embedding = torch.nn.Embedding(vocab_size, self.dim)
    
    self.initial_h = nn.ParameterList(
      [torch.nn.Parameter(torch.randn(1, self.num_heads, self.hidden_dim), requires_grad=True) for _ in range(self.depth)]
    )
    self.initial_c = nn.ParameterList(
      [torch.nn.Parameter(torch.randn(1, self.num_heads, self.hidden_dim), requires_grad=True) for _ in range(self.depth)]
    )
    
    # self.head_clock_denoms = torch.nn.Parameter(torch.tensor([.5]*self.num_heads))

  def forward(self, data, h, **kwargs):
    # print(f'{histories[0].shape=}, {hippo_c[0].shape=}')
    
    theta = self.angle_embedding(data) #shape: [batch_size of time t+1, dim]?
    
    accum = 0
    new_histories = []
    for layer_num in range(self.depth):
      
      output, history = self.limas[layer_num](theta+accum, h[layer_num])
      
      new_histories.append(history)
      accum += output.sum(dim=1) # sum across heads, shape [batch_size of time t+1, dim]

    predicts = self.predictor(accum, theta)

    scale_preact = self.scale_fc(torch.concat(new_histories, dim=2).sum(dim=1))
    scale = self.scale_act(scale_preact).sum(dim=1)

    return scale[:, None] * predicts, { 'h': new_histories }

  def initial_states(self):
    return {
      'h': [x for x in self.initial_h],
    }
  
  def embedding(self, input_ids):
    return self.predictor.embedding(input_ids)

  def all_word_embeddings(self):
    return self.predictor.all_word_embeddings()
  
  def pairwise_distance(self, A: torch.complex, B: torch.complex, distance='cos', eps=1e-8):
    """
    A: MxD, B: NxD
    distance: 'cos', 'l2'
    """
    if distance == 'cos':
      product = torch.einsum('md,nd->mn', A, B.conj())
      norms = A.norm(dim=1)[...,None] * B.norm(dim=1)[None,...] + eps
      return 1 - (product / norms).real
        
    if distance == 'l2':
      AT = A[..., None]
      BT = B.T[None, ...]
      # return (AT.real**2+AT.imag**2).sum(dim=0) + (BT.real**2+BT.imag**2).sum(dim=-1) - torch.einsum('DM...,...ND->MN', AT, BT.conj())
      return (AT - BT).norm(dim=1)
    
  # def inference(self, input_ids, histories, t, return_updated_history=False, distance='l2'):
  #   raise NotImplementedError # outdated
    
  #   # head_clocks = (1/self.head_clock_denoms[:, None]**(self.predictor.dimension_indices/self.dim))[:, None, :]
  #   batch = self.angle_embedding(input_ids)
  #   accum = 0  
  #   for lima_num, lima in enumerate(self.limas):
  #     # print(f'inference: {(batch-accum).shape}, {histories[lima_num].shape}')
  #     history = lima(batch - accum - histories[lima_num])
  #     histories[lima_num] = history# - head_clocks
  #     accum += history.sum(dim=0)
        
  #   predicts = self.predictor.forward(accum, input_ids, t+1, batch)
  #   pdist = self.pairwise_distance(predicts, self.predictor.all_word_embeddings(), distance=distance)
  #   minargs = pdist.argsort(dim=1)

  #   if return_updated_history:
  #     return minargs, histories
        
  #   return minargs