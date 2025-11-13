from typing import TypeAlias

import numpy as np
from jaxtyping import Int
from transformers import PreTrainedTokenizerBase

from absolute_bert.base_types import WordEmbeddings, Tokens
from absolute_bert.extractor import extract_multihead_semantics

from .models import AbsoluteAttention

QHeadConcepts: TypeAlias = Tokens["Dh num_tops"]
KHeadConcepts: TypeAlias = Tokens["Dh num_tops"]


def _convert_ids_to_tokens(
    id_matrix: Int[NDArray, "M N"], tokenizer: PreTrainedTokenizerBase
) -> Tokens["M N"]:
    return [tokenizer.convert_ids_to_tokens(ids) for ids in id_matrix]


def extract_attention_concepts(
    embeddings: WordEmbeddings, attention: AbsoluteAttention, tokenizer: PreTrainedTokenizerBase
) -> list[tuple[QHeadConcepts, KHeadConcepts]]:

    q_tops_lists: list[Int[NDArray, "Dh num_tops"]] = extract_multihead_concepts(
        embeddings, attention.Q.weight.detach(), attention.num_heads
    )
    k_tops_lists: list[Int[NDArray, "Dh num_tops"]] = extract_multihead_concepts(
        embeddings, attention.K.weight.detach(), attention.num_heads
    )

    q_token_lists = [_convert_ids_to_tokens(q_tops) for q_tops in q_tops_lists]
    k_token_lists = [_convert_ids_to_tokens(k_tops) for k_tops in k_tops_lists]

    return list(zip(q_token_lists, k_token_lists, strict=True))
