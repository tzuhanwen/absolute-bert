import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from dataclasses import dataclass
from absolute_bert.base_types import Labels, LanguageModel, States, Config
from .loss_types import Loss, LossForLMConfig
from .registry import LossType, loss_registry, loss_config_registry


@loss_registry.register(LossType.CROSS_ENTROPY)
class CrossEntropy(nn.Module):
    def __init__(self, model: LanguageModel, config: LossForLMConfig) -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = model

    def forward(self, outputs: States, labels: Labels) -> Loss:
        preds: Float[Tensor, "B T V"] = (
            outputs @ self.model.word_embeddings.T + self.model.word_biases
        )
        return self.loss(preds.view(-1, self.model.word_biases.shape[0]), labels.view(-1))


class CrossEntropyL2Embedding(nn.Module):
    def __init__(self, model, min_squared_norm=0, l2_squared_weight=1) -> None:
        super().__init__()
        self.model = model
        self.min_squared_norm = min_squared_norm
        self.l2_squared_weight = l2_squared_weight

        self.CE = nn.CrossEntropyLoss()

    def forward(self, predicts, targets, **kwargs):
        emb = self.model.all_word_embeddings()
        squared_norms = (emb.real**2 + emb.imag**2).sum(dim=1)

        return {
            "cross_entropy": self.CE(predicts, targets),
            "l2": self.l2_squared_weight
            * torch.clamp(squared_norms - self.min_squared_norm, min=0).mean(),
        }


@loss_config_registry.register(LossType.SAMPLED_SOFTMAX_CROSS_ENTROPY)
@dataclass
class LossConfig(Config):
    sampling_word_size: int = 100  # 用在 sampled softmax loss


class SampledSoftmaxCrossEntropy(nn.Module):
    """https://douglasorr.github.io/2021-10-training-objectives/3-sampled/article.html"""

    def __init__(self, model, num_sampling=100) -> None:
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
        reject_samples = (labels[..., None] == samples[None, :]) & (
            labels[..., None] != -100
        )  # 後面是 collator 會把非預測目標填為 -100
        noise_scores -= 1e6 * reject_samples
        noise_scores -= torch.log(
            (self.num_sampling - reject_samples.sum(-1, keepdims=True)).float()
        )

        # 5. Apply regular softmax cross entropy
        scores = torch.cat([label_scores[..., None], noise_scores], dim=-1)
        pseudo_label = torch.masked_fill(labels.clone(), labels != -100, 0).view(-1)
        loss = torch.nn.functional.cross_entropy(scores.view(-1, scores.shape[-1]), pseudo_label)

        return loss


class ComplexSampledSoftmaxCrossEntropy(nn.Module):
    """https://douglasorr.github.io/2021-10-training-objectives/3-sampled/article.html"""

    def __init__(self, model, num_sampling=100) -> None:
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
        noise_scores -= torch.log(
            (self.num_sampling - reject_samples.sum(-1, keepdims=True)).float()
        )

        # 5. Apply regular softmax cross entropy
        scores = torch.cat([label_scores[:, None], noise_scores], dim=1)
        loss = torch.nn.functional.cross_entropy(
            scores, torch.zeros(batch_size, dtype=torch.long).to(scores.device)
        )

        return {"loss": loss}
