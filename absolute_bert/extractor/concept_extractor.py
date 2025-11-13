from numpy.typing import NDArray
from jaxtyping import Float, Int
from sklearn.metrics.pairwise import cosine_similarity
from torch.types import Tensor

from absolute_bert.base_types import WordEmbeddings


def _extract_concepts(
    embeddings: WordEmbeddings, vectors: Float[Tensor, "N D"], num_tops: int = 5
) -> Int[NDArray, "N num_tops"]:

    sims: Float[NDArray, "V, N"] = cosine_similarity(embeddings, vectors)
    tops: Int[NDArray, "N num_tops"] = sims.argsort(axis=0)[::-1][:5].T

    return tops


def extract_multihead_concepts(
    embeddings: WordEmbeddings, 
    multihead: Float[Tensor, "H_Dh D"], 
    num_heads: int, 
    num_tops: int = 5,
) -> list[Int[NDArray, "Dh num_tops"]]:

    tops: Int[NDArray, "H_Dh num_tops"] = _extract_concepts(
        embeddings, multihead, num_tops=num_tops
    )

    return [
        tops[head_start_idx:head_start_idx+num_heads]
        for head_start_idx in range(0, len(tops), num_heads)
    ]
