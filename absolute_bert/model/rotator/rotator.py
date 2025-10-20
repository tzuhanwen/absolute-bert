import torch
from torch import nn

class Lima(nn.Module):
  def __init__(self, dim=64, eps=1e-4):
    super().__init__()
    self.eps = eps
    self.s = torch.nn.Parameter(torch.tensor(torch.pi*3/16+self.eps*7/4), requires_grad=False)
    self.a = torch.nn.Parameter(torch.tensor(torch.pi/4 - self.eps), requires_grad=False)
    self.b = torch.nn.Parameter(self.a.clone().detach()/4, requires_grad=False)

    self.dim = dim
    self.Theta = torch.nn.Parameter(torch.tensor(torch.rand(self.dim, self.dim))*torch.pi*2 - torch.pi, requires_grad=True)

    self.layer_norm = nn.LayerNorm(self.dim)

  def forward(self, theta):
    o = self.a - self.b + self.a*theta.cos() + self.b*(2*theta).cos()
    return self.layer_norm(o @ self.Theta)

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
      return torch.complex(self.embedding_real.weight, self.embedding_imag.weight)

  def forward(self, histories, sources):
    metric_theta = 1/self.rotary_denom**(self.dimension_nums/self.dim)
    total_angles = metric_theta + histories

    # cos, sin = total_angles.cos(), total_angles.sin()
    # embedding = self.embedding(sources)
    # real, imag = embedding[..., 0], embedding[..., 1]
    # return torch.stack([real*cos-imag*sin, real*sin+imag*cos], dim=-1)

    rotation = torch.complex(total_angles.cos(), total_angles.sin())
    return self.embedding(sources)*rotation

class Rotator(nn.Module):
  def __init__(self, vocab_size, dim=64, depth=3):
    super().__init__()
    self.dim = dim
    self.depth = depth
    self.limas = nn.ModuleList([Lima(dim=self.dim) for _ in range(self.depth)])
    self.initial_history = torch.nn.Parameter(torch.tensor(torch.randn(1, self.dim)), requires_grad=True)

    self.angle_embedding = torch.nn.Embedding(vocab_size, self.dim)
    self.predictor = Output(tokenizer.vocab_size)

  def embedding(self, input_ids):
    return self.predictor.embedding(input_ids)

  def forward(self, data, batch_sizes, **kwargs):

    history = self.initial_history

    for batch_accum, batch_size, next_batch_size in zip(itertools.accumulate(batch_sizes, initial=0), batch_sizes, batch_sizes[1:]):
      batch = data[batch_accum:batch_accum+batch_size][:next_batch_size]
      theta = self.angle_embedding(batch)
      # print(f"{data[batch_accum:batch_accum+batch_size].shape=}, {batch.shape}, {next_batch_size=}, {theta.shape=}")

      for lima_num, lima in enumerate(self.limas):
        diff = theta-history[:theta.shape[0], :]
        history = lima(diff)
        # print(f"In Rotator, {batch_accum=}, {lima_num=}, {diff=}")

      predicts = self.predictor(history, batch)
      labels = data[batch_accum+batch_size:batch_accum+batch_size+next_batch_size]

      yield predicts, self.embedding(labels)
    
  def pairwise_distance(self, A: torch.complex, B: torch.complex, distance='cos', eps=1e-8):
    """
    A: MxD, B: NxD
    distance: 'cos', 'L2'
    """
    if distance == 'cos':
      product = torch.einsum('md,nd->mn', A, B.conj())
      norms = A.norm(dim=1)[...,None] * B.norm(dim=1)[None,...] + eps
      return (product / norms).real
        
    if distance == 'L2':
      AT = A[..., None]
      BT = B.T[None, ...]
      # return (AT.real**2+AT.imag**2).sum(dim=0) + (BT.real**2+BT.imag**2).sum(dim=-1) - torch.einsum('DM...,...ND->MN', AT, BT.conj())
      return (AT - BT).norm(dim=1)
    
  def inference(self, input_ids, histories, return_updated_history=False, distance='cos'):
    for lima in self.limas:
        histories = lima(self.angle_embedding(input_ids) - histories)
        
    predicts = self.predictor.forward(histories, input_ids)
    pdist = self.pairwise_distance(predicts, self.predictor.all_word_embeddings(), distance=distance)
    minargs = pdist.argsort(dim=1)

    if return_updated_history:
        return minargs, histories
        
    return minargs