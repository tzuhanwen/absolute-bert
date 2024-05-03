import torch

def pairwise_distance(A: torch.complex, B: torch.complex, metric='l2', eps=1e-8):
  """
  A: M x D
  B: N x D
  distance: 'cos', 'l2'
  
  output: M x N
  """
  if metric == 'cos':
    product = torch.einsum('md,nd->mn', A, B.conj())
    norms = A.norm(dim=1)[...,None] * B.norm(dim=1)[None,...] + eps
    return 1 - (product / norms).real
      
  if metric == 'l2':
    AT = A[..., None]
    BT = B.T[None, ...]
    # return (AT.real**2+AT.imag**2).sum(dim=0) + (BT.real**2+BT.imag**2).sum(dim=-1) - torch.einsum('DM...,...ND->MN', AT, BT.conj())
    return (AT - BT).norm(dim=1)

def paired_distance(A: torch.complex, B: torch.complex, metric='l2', eps=1e-8):
  """
  A: [M, D]
  B: [M, D]
  distance: 'cos', 'l2'
  
  output: [M]
  """
  if metric == 'l2':
    return (A - B).norm(dim=1)
    
  if metric == 'cos':
    product = torch.einsum('md,md->m', A, B.conj())
    norms = A.norm(dim=1) * B.norm(dim=1) + eps
    return 1 - (product / norms).real