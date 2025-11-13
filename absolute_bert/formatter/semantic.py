from typing import TypeAlias

from absolute_bert.base_types import Tokens

# a concept is represented by <num_tops> related words
Concept: TypeAlias = list[str]
Concepts: TypeAlias = Tokens["* num_tops"]
ConceptsTuple: TypeAlias = tuple[Concepts, ...]

# semantic: A formatted multiline string that visualizes the attention projections.
# It is printed as several blocks (e.g., Q / K / V), and each block contains
# multiple lines of word-level tokens describing the projection’s operations.
SemanticStr: TypeAlias = str


def _get_concepts_width(concepts: Concepts) -> int:
    return max([len(", ".join(concept)) for concept in concepts])


def _pad_lines(concepts_tuple: ConceptsTuple) -> ConceptsTuple:
    num_conceptcs = [len(concepts) for concepts in concepts_tuple]
    max_height = max(num_concepts)
    return tuple(
        concepts + [[]] * (max_height - height)
        for height, concepts in zip(num_concepts, concepts_tuple)
    )


def to_semantic_str(concepts_tuple: ConceptsTuple) -> SemanticStr:

    space_between_concept_block = 2
    concept_block_widths = [_get_concepts_width(concepts) for concepts in concepts_tuple]
    full_block_widths = [
        width + space_between_concept_block + 2  # 2 for [] bracket
        for width in concept_block_widths
    ]

    height_padded_concepts_tuple = _pad_lines(concepts_tuple)

    lines = []
    for nth_concept_tuple in zip(*height_padded_concepts_tuple):

        line = ""
        for width, concept in zip(full_block_widths, nth_concept_tuple):
            concept_str = f"[{', '.join(concept)}]"
            line += concept_str + " " * (width - len(concept_str))

        lines.append(line)

    return "\n".join(lines) + "\n"
