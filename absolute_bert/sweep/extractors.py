from typing import TypeAlias

from torch.types import Device
from transformers import PreTrainedTokenizerBase

from absolute_bert.model.absolute_bert.concept_extractor import extract_attention_concepts
from absolute_bert.model.absolute_bert.models import AbsoluteBertLM
from absolute_bert.formatter.semantic import to_semantic_str, SemanticStr

ModuleName: TypeAlias = str
HeadSemantic: TypeAlias = SemanticStr
ModuleSemantic: TypeAlias = list[SemanticStr]
NamedModuleSemantic: TypeAlias = tuple[ModuleName, ModuleSemantic]


def _extract_absolute_bert_semantics(
    lm: AbsoluteBertLM, tokenizer: PreTrainedTokenizerBase, device: Device = "cpu"
) -> list[NamedModuleSemantic]:

    embeddings = lm.word_embeddings

    layer_semantics: list[NamedSemantic] = []
    for layer_num, layer in enumerate(lm.base_model.layers):
        head_concepts_tuples = extract_attention_concepts(
            embeddings, layer.attention, tokenizer
        )

        attention_semantic: ModuleSemantic = []
        for head_concepts_tuple in head_concepts_tuples:
            attention_semantic.append(to_semantic_str(head_concepts_tuple))

        layer_semantics.append((f"layer_{layer_num}", attention_semantic))

    return layer_semantics


LayerName: TypeAlias = str


def get_absolute_bert_semantic_summary(
    lm: AbsoluteBertLM, tokenizer: PreTrainedTokenizerBase, device: Device = "cpu"#, sampling_head: int | None = None
) -> dict[LayerName, str]:
    
    layer_semantics: list[NamedModuleSemantic] = _extract_absolute_bert_semantics(
        lm, tokenizer, device
    )

    layer_blocks: dict[LayerName, str] = {}

    return {
        layer_name: "\n".join(layer_semantic)
        for layer_name, layer_semantic in layer_semantics
    }