import numpy as np
import torch
import torch.nn as nn

class Absolute_attention(nn.Module):
  def __init__(self, dim=256, num_heads=16, hidden_dim=None):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads

    if isinstance(hidden_dim, int):
      self.hidden_dim = hidden_dim
    else:
      assert dim % num_heads == 0
      self.hidden_dim = dim // num_heads
      
    # assert self.hidden_dim % 2 == 0
    # self.time_embedding = time_embedding
    self.time_angle = nn.Parameter(torch.rand(self.hidden_dim))
    self.head_time_delta = nn.Parameter(torch.rand(self.num_heads))

    self.Q = nn.Linear(dim, num_heads * (self.hidden_dim + 1), bias=False)
    self.K = nn.Linear(dim, num_heads * self.hidden_dim, bias=False)
    self.V = nn.Linear(dim, num_heads * self.hidden_dim, bias=False)
    self.O = nn.Linear(num_heads * self.hidden_dim, dim)
    
  def forward(self, tensor, attention_mask):
    batch_length = tensor.shape[:2]
    
    q = self.Q(tensor).view(*batch_length, self.num_heads, self.hidden_dim + 1)
    # q = (q + attention_mask[..., None, None])
    q = q.softmax(dim=-1)
    
    time_angles = (torch.arange(batch_length[1]).to(self.head_time_delta.device)[:, None, None] 
                   + self.head_time_delta[None, :, None]) * self.time_angle
    cosines, sines = time_angles.cos(), time_angles.sin()
    time = torch.cat([cosines + sines, cosines - sines], dim=-1) / np.sqrt(self.hidden_dim)
    q = (1 - q[..., -1, None]) * time

    k_time_angles = torch.arange(batch_length[1]).to(self.time_angle.device)[:, None, None] * self.time_angle
    k_cosines, k_sines = time_angles.cos(), time_angles.sin()
    k_time = torch.cat([k_cosines + k_sines, k_cosines - k_sines], dim=-1) / np.sqrt(self.hidden_dim)
    qk_weight = torch.einsum('blhd,lhd->blh', q, k_time)

    k = self.K(tensor).view(*batch_length, self.num_heads, self.hidden_dim)
    k = k.softmax(dim=-1) * attention_mask[..., None, None]
    
    whole_attention = qk_weight[..., None] * k

    v = self.V(tensor)
    output = self.O(whole_attention.reshape(*batch_length, -1) * v)
    
    return output

class Absolute_bert(nn.Module):
  def __init__(self, vocab_size, dim=256, num_heads=8, hidden_dim=None, depth=8, dtype=torch.float):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.depth = depth
    
    self.embedding = nn.Embedding(vocab_size, dim, _weight=torch.nn.init.xavier_normal_(torch.ones([vocab_size, self.dim])))
    self.layers = nn.ModuleList([Absolute_attention(dim=dim, 
                                                    num_heads=num_heads,
                                                    hidden_dim=hidden_dim
                                                   ) for _ in range(depth)])
    self.dtype = dtype

  def forward(self, input_ids, attention_mask, **kwargs):
    # extended_attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
    
    tensor = self.embedding(input_ids)
    
    for layer in self.layers:
      output = layer(tensor, attention_mask)
      tensor = tensor + output * attention_mask[..., None]
    
    return tensor

class Absolute_bert_for_masked_LM(nn.Module):
  def __init__(self, vocab_size, dim=256, num_heads=8, hidden_dim=None, depth=8):
    super().__init__()
    self.base_model = Absolute_bert(vocab_size, dim, num_heads, hidden_dim, depth)
    self.bias = nn.Parameter(torch.zeros(vocab_size))

  def forward(self, input_ids, attention_mask, labels, **kwargs):
    tensor = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    return tensor, labels

  def word_embeddings(self):
    return self.base_model.embedding.weight