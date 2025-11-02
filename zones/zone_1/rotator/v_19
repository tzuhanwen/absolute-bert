# 最主要的差別就是 history 也會被 rotate

import torch
from torch import nn
import itertools

class Multihead_Lima(nn.Module):
  def __init__(self, dim=64, hidden_dim=32, history_dim_in=16, history_dim_out=None, num_heads=4, fix_max_value=1, min_proportion=0.2):
    super().__init__()
    
    self.dim = dim
    self.hidden_dim = hidden_dim
    self.history_dim_in = history_dim_in
    self.history_dim_out = hidden_dim if (history_dim_out==None) else history_dim_out
    self.num_heads = num_heads
      
    if fix_max_value: 
      assert fix_max_value > 0
      self.M = torch.nn.Parameter(torch.tensor(fix_max_value), requires_grad=False)
    else:
      raise NotImplementedError

    assert min_proportion > 0 and min_proportion <= 1
    self.min_proportion = min_proportion
    self.proportion = torch.nn.Parameter(torch.rand(self.num_heads, self.hidden_dim)) # weight 的 range 會限制在 M 下面的一個比例: 1/(e^{proportion}+1)
    self.lima_shape = torch.nn.Parameter(torch.rand(self.num_heads, self.hidden_dim)) # 類似之前的 b/a, 會用 sigmoid 把 range 限制在 -0.25 ~ 0.25
    self.fc_in = torch.nn.Linear(self.num_heads*(self.dim+self.history_dim_in), self.num_heads*self.hidden_dim)
    self.Theta = torch.nn.Parameter(torch.rand(self.num_heads, self.hidden_dim, self.history_dim_out)*torch.pi*2 - torch.pi, requires_grad=True)
    self.fc_out = torch.nn.Linear(self.num_heads*self.history_dim_out, self.num_heads*self.dim)
    self.layer_norm = nn.LayerNorm(self.history_dim_out, bias=False)

  def forward(self, theta, histories):
    """
    theta: shape [batch_size, dim_in]
    histories: shape [batch_size, num_heads, history_dim]
    """
    batch_size = theta.shape[0]
    total_in = torch.concat([theta[:, None, :].expand(-1, self.num_heads, -1), histories], dim=-1).view(batch_size, -1)
    h = self.fc_in(total_in).view(batch_size, self.num_heads, self.hidden_dim)
    
    b_a = ((torch.sigmoid(self.lima_shape) - 0.5) / 2)  # b/a, shape: [num_heads, hidden_dim]
    scale_controlling_coefficient = self.M * (1-self.min_proportion) / (2*(torch.pow(torch.e, self.proportion)+1)) # shape: [num_heads, hidden_dim]
    o = self.M + scale_controlling_coefficient * (-1 - b_a + h.cos() + b_a*(2*h).cos())
    # print(f'multihead_lima: {o.shape=}, {self.Theta.shape=}')
    raw_resulting_angle = torch.einsum('bhm,hmn->bhn', o, self.Theta) #shape: [batch_size, num_heads, hidden_dim]
    
    next_histories = self.layer_norm(raw_resulting_angle)
    # next_histories = raw_resulting_angle
    output = self.fc_out(raw_resulting_angle.reshape(batch_size, -1)).view(batch_size, self.num_heads, self.dim)
    return output, next_histories

class Output(nn.Module):
  def __init__(self, vocab_size, dim=64, rotary_denom=.5, local_coordinating=True, **kwargs):
    super().__init__()
    self.dim = dim
    self.vocab_size = vocab_size
    self.dimension_indices = torch.nn.Parameter(torch.arange(0, dim, dtype=torch.float), requires_grad=False)
    self.initial_rotary_denom = rotary_denom  # 參考 transformer 的 position embedding 分母的 10000
    self.log_rotary_denom = torch.nn.Parameter(torch.tensor(self.initial_rotary_denom, dtype=torch.float))
    self.embedding_real = torch.nn.Embedding(vocab_size, self.dim)
    self.embedding_imag = torch.nn.Embedding(vocab_size, self.dim)
    self.local_coordinating = local_coordinating

  def embedding(self, input_ids):
    return torch.complex(self.embedding_real(input_ids), self.embedding_imag(input_ids))
    # return torch.stack([self.embedding_real(input_ids), self.embedding_imag(input_ids)], dim=-1)

  def all_word_embeddings(self):
      return torch.complex(self.embedding_real.weight, self.embedding_imag.weight).clone()

  def forward(self, angles, sources, word_angles=None):
    time_angle = 1 / (torch.e**self.log_rotary_denom)**(self.dimension_indices/self.dim) # shape: [1,dim]
    total_angles = time_angle + angles
    if self.local_coordinating:
      total_angles += word_angles
    rotation = torch.complex(total_angles.cos(), total_angles.sin())
    return self.embedding(sources)*rotation

class Hippo(nn.Module):
  def __init__(self, dim, c_dim=32, num_heads=4):
    super().__init__()
    self.dim = dim
    self.c_dim = c_dim
    self.num_heads = num_heads

    q = torch.arange(0, c_dim, dtype=torch.float)
    B = torch.pow(2*q+1, .5) # shape [1, c_dim]
    A = (B * B[:, None]).tril().fill_diagonal_(0) + torch.diag(q+1)
    
    self.AT = nn.Parameter(A.T, requires_grad=False) # transpose of a lower triangular matrix
    self.B = nn.Parameter(B, requires_grad=False)
    self.Lf = nn.Linear(self.dim, self.num_heads)
    self.eye = nn.Parameter(torch.eye(self.c_dim), requires_grad=False)

  def forward(self, c, h, t):
    """
    c: shape [batch_size, num_heads, num_hippo_heads*c_dim]
    h: shape [batch_size, num_heads, dim]
    """
    batchSize_numLimaHeads = c.shape[:2]
    f = self.Lf(h)[..., None] # shape [batch_size, num_heads, num_hippo_heads, 1]
    A_part = torch.einsum('bhjm,mn', c.view(*batchSize_numLimaHeads, self.num_heads, self.c_dim), self.eye - self.AT/t)

    return (A_part + (f * self.B)/t).view(*batchSize_numLimaHeads, -1)

  def reconstruct(self, c, density=50):
    """
    x: shape [n, 1]
    """
    vals = np.linspace(0.0, 1.0, density)
    eval_matrix = self.B[:, None] * ss.eval_legendre(np.arange(self.c_dim)[:, None], 2*vals-1).astype(np.float32)
    # print(eval_matrix)
    return c @ eval_matrix

class Rotator(nn.Module):
  def __init__(self, vocab_size, dim=64, hidden_dim=32, history_dim=None, hippo_dim=16, num_hippo_heads=4, num_heads=4, depth=3, rotary_denom=.5, **kwargs):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.hidden_dim = hidden_dim
    self.history_dim = hidden_dim if (history_dim==None) else history_dim 
    self.hippo_dim = hippo_dim
    self.num_hippo_heads = num_hippo_heads
    self.num_heads = num_heads
    self.depth = depth
    
    self.limas = nn.ModuleList([Multihead_Lima(
      dim = self.dim,
      history_dim_in = self.history_dim + self.num_hippo_heads*self.hippo_dim,
      history_dim_out = self.history_dim,
      hidden_dim = self.hidden_dim,
      num_heads = self.num_heads
    ) for _ in range(self.depth)])
    self.hippos = nn.ModuleList([
      Hippo(dim=self.history_dim, c_dim=self.hippo_dim, num_heads=self.num_hippo_heads)
    for _ in range(self.depth)])
    self.predictor = Output(vocab_size, dim=self.dim, rotary_denom=rotary_denom, **kwargs)
    self.angle_embedding = torch.nn.Embedding(self.vocab_size, self.dim)
    
    self.initial_history = nn.ParameterList(
      [torch.nn.Parameter(torch.randn(1, self.num_heads, self.hidden_dim), requires_grad=True) for _ in range(self.depth)]
    )
    self.initial_c = nn.Parameter(torch.zeros([1, 1, self.num_hippo_heads*self.hippo_dim]), requires_grad=False)
    
    # self.head_clock_denoms = torch.nn.Parameter(torch.tensor([.5]*self.num_heads))

  def embedding(self, input_ids):
    return self.predictor.embedding(input_ids)

  def all_word_embeddings(self):
    return self.predictor.all_word_embeddings()

  def initial_states(self):
    histories = [history for history in self.initial_history] #copying ref list # 近期記憶
    hippo_c = [self.initial_c.expand(-1, self.num_heads, -1) for _ in range(self.depth)] # 整體記憶
    return {
      'histories': histories,
      'hippo_c': hippo_c
    }
  
  def forward(self, data, time, histories, hippo_c, **kwargs):
    # print(f'{histories[0].shape=}, {hippo_c[0].shape=}')
    
    theta = self.angle_embedding(data) #shape: [batch_size of time t+1, dim]?
    
    accum = 0
    new_hippo_c, new_histories = [], []
    for layer_num in range(self.depth):
      # print(f'{histories[layer_num].shape=}, {hippo_c[layer_num].shape=}')
      histories_hippos = torch.concat([histories[layer_num], hippo_c[layer_num]], dim=-1) # shape: [batch_size of time t+1, num_heads, dim]
      # print(f'{histories_hippos.shape=}')
      output, history = self.limas[layer_num](theta-accum, histories_hippos)
      # print(f'{history.shape=}')
      
      new_hippo_c.append(self.hippos[layer_num](hippo_c[layer_num], history, time))
      new_histories.append(history)
      accum += output.sum(dim=1) # sum across heads, shape [batch_size of time t+1, dim]

      predicts = self.predictor(accum, data, theta)

    return predicts, { 'histories': new_histories, 'hippo_c': new_hippo_c }
    
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
    
  def inference(self, input_ids, histories, t, return_updated_history=False, distance='l2'):
    raise NotImplementedError # outdated
    
    # head_clocks = (1/self.head_clock_denoms[:, None]**(self.predictor.dimension_indices/self.dim))[:, None, :]
    batch = self.angle_embedding(input_ids)
    accum = 0  
    for lima_num, lima in enumerate(self.limas):
      # print(f'inference: {(batch-accum).shape}, {histories[lima_num].shape}')
      history = lima(batch - accum - histories[lima_num])
      histories[lima_num] = history# - head_clocks
      accum += history.sum(dim=0)
        
    predicts = self.predictor.forward(accum, input_ids, t+1, batch)
    pdist = self.pairwise_distance(predicts, self.predictor.all_word_embeddings(), distance=distance)
    minargs = pdist.argsort(dim=1)

    if return_updated_history:
      return minargs, histories
        
    return minargs