import numpy as np
import torch
import torch.nn as nn

from . import distances

class Complex_mse(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

  def __repr__(self):
      return f'mse'

  def forward(self, predicts, targets):
    diffs = (predicts - targets)
    # diff_square_norms = diffs * diffs.conj()
    diff_square_norms = diffs.real**2 + diffs.imag**2
    return {
        'mse': diff_square_norms.real.sum(dim=[1]).mean()
    }

class Complex_mse_with_inverse_norm(Complex_mse):
    def __init__(self, relax=.25, min_dist=1, **kwargs):
        super().__init__()
        self.relax = relax
        self.min_dist = min_dist
        
    def __repr__(self):
        return f'mse_with_inverse_norm(relax={self.relax}, min_dist={self.min_dist})'

    def forward(self, predicts, targets):
        inverse_norm = torch.clamp(1/(predicts.norm(dim=-1)+self.relax) - 1/(self.min_dist+self.relax), min=0)
        return  {
            **super().forward(predicts, targets),
            'inverse_norm': inverse_norm.mean()
        }

class Complex_triplet_loss(nn.Module):
  def __init__(self, model, sampling_word_size=10, margin=5, distance_metric='l2', **kwargs):
    super().__init__()
    self.sampling_word_size = sampling_word_size
    self.distance_metric = distance_metric
    self.margin = margin
    self.model = model

  def __repr__(self):
      return f'triplet_loss(margin={self.margin})'

  def forward(self, predicts, targets):
    sampled_word_vecs = self.model.predictor.all_word_embeddings()[np.random.choice(self.model.vocab_size, size=self.sampling_word_size)]

    pos_dist = distances.paired_distance(predicts, targets, metric=self.distance_metric)
    sampled_dists = distances.pairwise_distance(targets, sampled_word_vecs, metric=self.distance_metric)
    neg_dist = sampled_dists.min(dim=1).values
      
    return {
        'triplet_loss': torch.clamp(pos_dist-neg_dist+self.margin, min=0).mean()
    }

class Complex_multiplet_loss(nn.Module):
  def __init__(self, model, sampling_word_size=10, margin=5, distance_metric='l2', **kwargs):
    super().__init__()
    self.sampling_word_size = sampling_word_size
    self.distance_metric = distance_metric
    self.margin = margin
    self.model = model

  def __repr__(self):
      return f'multiplet_loss(margin={self.margin}, sampling_word_size={self.sampling_word_size})'

  def forward(self, predicts, targets):
    sampled_word_vecs = self.model.predictor.all_word_embeddings()[np.random.choice(self.model.vocab_size, size=self.sampling_word_size)]

    pos_dist = distances.paired_distance(predicts, targets, metric=self.distance_metric) #shape: [batch]
    sampled_dists = distances.pairwise_distance(targets, sampled_word_vecs, metric=self.distance_metric) 
    triplets = (pos_dist[:, None] - sampled_dists + self.margin) # shape: [batch, self.sampling_word_size]
    
    return {
        'multiplet_loss': torch.clamp(triplets, min=0).mean()
    }

        
class Complex_mse_triplet_loss(nn.Module):
  def __init__(self, model, sampling_word_size=10, margin=5, distance_metric='l2', **kwargs):
    super().__init__()
    self.triplet = Complex_triplet_loss(model, sampling_word_size, margin, distance_metric)
    self.mse = Complex_mse()

  def __repr__(self):
      return f'{str(self.triplet)}+{str(self.mse)}'

  def forward(self, predicts, targets):
    return {
      **self.triplet(predicts, targets),
      **self.mse(predicts, targets)
    }

class Complex_mse_squared_triplet_loss(nn.Module):
  def __init__(self, model, sampling_word_size=10, margin=5, distance_metric='l2', **kwargs):
    super().__init__()
    self.triplet = Complex_triplet_loss(model, sampling_word_size, margin, distance_metric)
    self.mse = Complex_mse()

  def __repr__(self):
      return f'sq{str(self.triplet)}+{str(self.mse)}'

  def forward(self, predicts, targets):
    return {
      'sqtriplet_loss': self.triplet(predicts, targets)['triplet_loss']**2,
      **self.mse(predicts, targets)
    }

class Mse_multiplet_loss(nn.Module):
  def __init__(self, model, sampling_word_size=10, margin=5, distance_metric='l2', **kwargs):
    super().__init__()
    self.multiplet = Complex_multiplet_loss(model, sampling_word_size, margin, distance_metric)
    self.mse = Complex_mse()

  def __repr__(self):
      return f'{str(self.multiplet)}+{str(self.mse)}'

  def forward(self, predicts, targets):
    return {
      **self.multiplet(predicts, targets),
      **self.mse(predicts, targets)
    }

class Cos_multiplet_loss(nn.Module):
  def __init__(self, model, sampling_word_size=10, margin=5, eps=1e-8, **kwargs):
    super().__init__()
    self.multiplet = Complex_multiplet_loss(model, sampling_word_size, margin, distance_metric='cos')
    self.eps = eps

  def __repr__(self):
    return f'{str(self.multiplet)}+cos_dist'

  def forward(self, predicts, targets):
    return {
      **self.multiplet(predicts, targets),
      'cos': distances.paired_distance(predicts, targets, metric='cos', eps=self.eps).mean()
    }

class Cross_entropy_l2_embedding(nn.Module):
  def __init__(self, model, min_squared_norm=0, l2_squared_weight=1):
    super().__init__()
    self.model = model
    self.min_squared_norm = min_squared_norm
    self.l2_squared_weight = l2_squared_weight

    self.CE = nn.CrossEntropyLoss()

  def forward(self, predicts, targets, **kwargs):
    emb = self.model.all_word_embeddings()
    squared_norms = (emb.real**2 + emb.imag**2).sum(dim=1)
    
    return {
      'cross_entropy': self.CE(predicts, targets),
      'l2': self.l2_squared_weight * torch.clamp(squared_norms-self.min_squared_norm, min=0).mean()
    }

class Sampled_softmax_cross_entropy(nn.Module):
  """https://douglasorr.github.io/2021-10-training-objectives/3-sampled/article.html"""
  
  def __init__(self, model, num_sampling=100):
    super().__init__()
    self.num_sampling = num_sampling
    self.model = model

  def forward(self, predictions, labels):
    """
    preds: shape [batch_size, dim]
    labels: shape [batch_size, dim]
    """
    batch_size = predictions.shape[0]
    predictions = torch.cat([predictions.real, predictions.imag], dim=-1)
    
    # model = ...  # returns (batch_size x embedding_size)
    projection = self.model.model.all_word_embeddings()  # shape (n_classes x embedding_size)
    projection = torch.cat([projection.real, projection.imag], dim=-1)
    
    n_classes = projection.shape[0]
    
    # 2. Get target label scores, paired_inner_product(pred_emb, label_emb)
    label_scores = (predictions * projection[labels, :]).sum(-1) + self.model.bias[labels]
    
    # 3. Sample shared noise & get scores
    samples = torch.randint(high=n_classes, size=[self.num_sampling]).to(labels.device)
    noise_scores = predictions @ projection[samples, :].T + self.model.bias[None, samples]
    noise_scores += np.log(n_classes - 1)
    
    # 4. Reject samples matching target label & correct for remaining samples
    reject_samples = labels[:, None] == samples[None, :]
    noise_scores -= 1e6 * reject_samples
    noise_scores -= torch.log((self.num_sampling - reject_samples.sum(-1, keepdims=True)).float())
    
    # 5. Apply regular softmax cross entropy
    scores = torch.cat([label_scores[:, None], noise_scores], dim=1)
    loss = torch.nn.functional.cross_entropy(scores, torch.zeros(batch_size, dtype=torch.long).to(scores.device))

    return { 'loss': loss }