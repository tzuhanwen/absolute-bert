import torch
import torch.nn as nn
import numpy as np

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
    # batch_sizes = predictions.shape[:-1]

    # model = ...  # returns (batch_size x embedding_size)
    projection = self.model.word_embeddings()  # shape (n_classes x embedding_size)
    n_classes = projection.shape[0]

    # 2. Get target label scores, paired_inner_product(pred_emb, label_emb)
    label_scores = (predictions * projection[labels, :]).sum(-1) + self.model.bias[labels]

    # 3. Sample shared noise & get scores
    samples = torch.randint(high=n_classes, size=[self.num_sampling]).to(labels.device)
    noise_scores = predictions @ projection[samples, :].T + self.model.bias[None, samples]
    noise_scores += np.log(n_classes - 1)

    # 4. Reject samples matching target label & correct for remaining samples
    reject_samples = (labels[..., None] == samples[None, :]) & (labels[..., None] != -100)  #後面是 collator 會把非預測目標填為 -100
    noise_scores -= 1e6 * reject_samples
    noise_scores -= torch.log((self.num_sampling - reject_samples.sum(-1, keepdims=True)).float())

    # 5. Apply regular softmax cross entropy
    scores = torch.cat([label_scores[..., None], noise_scores], dim=-1)
    pseudo_label = torch.masked_fill(labels.clone(), labels != -100, 0).view(-1)
    loss = torch.nn.functional.cross_entropy(scores.view(-1, scores.shape[-1]), pseudo_label)

    return loss


class Cross_entropy(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.loss = torch.nn.CrossEntropyLoss()
    self.model = model

  def forward(self, outputs, labels):
    preds = outputs @ self.model.word_embeddings().T + self.model.bias
    return self.loss(preds.view(-1, self.model.bias.shape[0]), labels.view(-1))