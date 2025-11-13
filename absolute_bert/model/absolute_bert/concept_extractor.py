from typing import TypeAlias

import numpy as np
from jaxtyping import Int
from numpy.typing import NDArray
from torch.types import Device
from transformers import PreTrainedTokenizerBase

from absolute_bert.base_types import WordEmbeddings, Tokens
from absolute_bert.extractor import extract_multihead_concepts

from .models import AbsoluteAttention

QHeadConcepts: TypeAlias = Tokens["Dh num_tops"]
KHeadConcepts: TypeAlias = Tokens["Dh num_tops"]


def _convert_ids_to_tokens(
    id_matrix: Int[NDArray, "M N"], tokenizer: PreTrainedTokenizerBase
) -> Tokens["M N"]:
    return [tokenizer.convert_ids_to_tokens(ids) for ids in id_matrix]


def extract_attention_concepts(
    embeddings: WordEmbeddings, 
    attention: AbsoluteAttention, 
    tokenizer: PreTrainedTokenizerBase, 
    device: Device = "cpu"
) -> list[tuple[QHeadConcepts, KHeadConcepts]]:

    embeddings = embeddings.detach().to(device)

    q_tops_lists: list[Int[NDArray, "Dh num_tops"]] = extract_multihead_concepts(
        embeddings, attention.Q.weight.detach().to(device), attention.num_heads
    )
    k_tops_lists: list[Int[NDArray, "Dh num_tops"]] = extract_multihead_concepts(
        embeddings, attention.K.weight.detach().to(device), attention.num_heads
    )

    q_token_lists = [_convert_ids_to_tokens(q_tops, tokenizer) for q_tops in q_tops_lists]
    k_token_lists = [_convert_ids_to_tokens(k_tops, tokenizer) for k_tops in k_tops_lists]

    return list(zip(q_token_lists, k_token_lists, strict=True))
