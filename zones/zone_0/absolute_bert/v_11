import numpy as np
import torch
import torch.nn as nn

class Absolute_attention(nn.Module):
  def __init__(self, dim=512, num_heads=8, hidden_dim=None, time_dim=64, linear_dim=3*512, k_temperature=.5):
    super().__init__()

    self.dim = dim
    self.num_heads = num_heads
    self.time_dim = time_dim
    self.linear_dim = linear_dim

    if isinstance(hidden_dim, int):
      self.hidden_dim = hidden_dim
    else:
      assert dim % num_heads == 0
      self.hidden_dim = dim // num_heads

    # assert self.hidden_dim % 2 == 0
    # self.time_embedding = time_embedding
    self.time_angle = nn.Parameter(torch.rand(self.num_heads, self.time_dim))
    self.head_time_delta = nn.Parameter(torch.rand(self.num_heads))

    self.Q = nn.Linear(dim, num_heads * self.hidden_dim, bias=False)
    self.K = nn.Linear(dim, num_heads * self.hidden_dim)
    self.V = nn.Linear(dim, num_heads * self.hidden_dim)
    self.O = nn.Linear(num_heads * self.hidden_dim, dim)

    self.q_bias = nn.Parameter(torch.zeros(num_heads * self.hidden_dim))
    self.k_temperature = k_temperature

    with torch.no_grad():
      self.V.bias.copy_(torch.zeros_like(self.V.bias))
      self.O.bias.copy_(torch.zeros_like(self.O.bias))
    
    self.dropout = nn.Dropout(p=0.5)
    self.layer_norm = nn.LayerNorm(dim)

    self.linear_in = nn.Linear(self.dim, self.linear_dim)
    self.act_fn = nn.GELU(approximate='tanh')
    self.linear_out = nn.Linear(self.linear_dim, self.dim)
    self.layer_norm_lin = nn.LayerNorm(dim)

  def forward(self, tensor, attention_mask):
    batch_length = tensor.shape[:2]
    # print(f'{tensor.shape=}')
    q = (self.Q(tensor) - self.q_bias.exp()).view(*batch_length, self.num_heads, self.hidden_dim)
    # q = (q + attention_mask[..., None, None])
    q = (q.sigmoid() / self.hidden_dim) * attention_mask[..., None, None]
    
    time_angles = (torch.arange(batch_length[1]).to(self.head_time_delta.device)[:, None, None]
                   + self.head_time_delta[None, :, None]) * self.time_angle  # shape: [length, num_heads, dim_time] ?
    cosines, sines = time_angles.cos(), time_angles.sin()
    time = torch.cat([cosines + sines, cosines - sines], dim=-1) / np.sqrt(self.hidden_dim)  # shape: [length, num_heads, 2*dim_time] ?
    q = q.sum(-1)[..., None] * time
    # q shape: [batch_size, length, num_heads, 2*dim_time] = [batch_size, length, 1, 1] * [length, num_heads, 2*dim_time] ?

    k = (self.K(tensor)).view(*batch_length, self.num_heads, self.hidden_dim)  # bthd
    k = (k / self.k_temperature).softmax(dim=-1) * attention_mask[..., None, None]

    k_time_angles = torch.arange(batch_length[1]).to(self.time_angle.device)[:, None, None] * self.time_angle
    k_cosines, k_sines = time_angles.cos(), time_angles.sin()
    k_time = torch.cat([k_cosines + k_sines, k_cosines - k_sines], dim=-1) / np.sqrt(self.hidden_dim)

    attention = torch.einsum('blhd,thd->blth', q, k_time)

    v = self.V(tensor).view(*batch_length, self.num_heads, self.hidden_dim)
    adding_comb = torch.einsum('blth,bthd->blhd', attention, v)

    attention_output = self.O(adding_comb.reshape(*batch_length, self.dim))
    intermediate = self.layer_norm(tensor + self.dropout(attention_output))

    lin_output = self.dropout(self.linear_out(self.act_fn(self.linear_in(intermediate))))

    return self.layer_norm_lin(intermediate + lin_output)

class Absolute_bert(nn.Module):
  def __init__(self, vocab_size, dim=512, num_heads=8, hidden_dim=None, linear_dim=3*512, depth=8, log_granularity=[6,6,6,6,6,6,6,6], attention_type=Absolute_attention, dtype=torch.float, **kwargs):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.linear_dim = linear_dim
    self.depth = depth
    assert len(log_granularity) == depth
    self.log_granularity = log_granularity

    self.embedding = nn.Embedding(vocab_size, dim, _weight=torch.nn.init.xavier_normal_(torch.ones([vocab_size, self.dim])))
    # self.embedding = nn.Embedding(vocab_size, dim, _weight=torch.rand(vocab_size, self.dim) / np.sqrt(self.dim))

    self.layers = nn.ModuleList([Absolute_attention(dim=dim,
      num_heads=dim//(2**gran),
      hidden_dim=2**gran,
      **kwargs
      ) for gran in log_granularity])
    
    self.dtype = dtype

  def forward(self, input_ids, attention_mask, **kwargs):
    # extended_attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

    tensor = self.embedding(input_ids)

    for layer in self.layers:
      tensor = layer(tensor, attention_mask)

    return tensor

class Absolute_bert_for_masked_LM(nn.Module):
  def __init__(self, vocab_size, dim=256, num_heads=8, hidden_dim=None, depth=8, **kwargs):
    super().__init__()
    self.base_model = Absolute_bert(vocab_size, dim, num_heads, hidden_dim, depth, **kwargs)
    self.bias = nn.Parameter(torch.zeros(vocab_size))

  def forward(self, input_ids, attention_mask, labels=None, **kwargs):
    tensor = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    return tensor, labels

  def word_embeddings(self):
    return self.base_model.embedding.weight