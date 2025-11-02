from abc import ABC, abstractmethod
import datasets
from datasets import load_dataset
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, BatchEncoding

from typing import TypedDict, NotRequired
from enum import Enum
from dataclasses import dataclass


@dataclass
class Dataset:
    path: str
    name: str | None
    extracting_part: str


# https://github.com/huggingface/datasets/blob/main/src/datasets/load.py
dataset_dict = {
    "bookcorpus": Dataset("bookcorpus", None, "text"),
    "ms_marco": Dataset("ms_marco", "v2.1", "query"),
    "wikipedia": Dataset("wikipedia", "20220301.en", "text"),
}


class Trainloader:
    def __init__(self, dataset, batch_size) -> None:
        self.data = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + 1

    def __iter__(self):
        current_start = 0
        for i in range(self.__len__()):
            yield self.data[current_start : current_start + self.batch_size]
            current_start += self.batch_size


class Corpus(ABC):
    def __init__(
        self,
        corpus_name: str,
        tokenizer: PreTrainedTokenizerBase,
        cache_dir: str | None,
        batch_size: int = 128,
        max_length: int = 128,
        num_chunks: int = 4,
        shuffle=False,
    ) -> None:
        "Extract the first <max_length>*<num_chunks> tokens of each text in the corpus, and split into <num_chunks> data."

        self.corpus_name = corpus_name
        self.tokenizer = tokenizer

        dataset = dataset_dict[corpus_name]
        self.dataset = load_dataset(
            dataset.path,
            dataset.name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.max_length = max_length
        assert max_length % num_chunks == 0
        self.num_chunks = num_chunks

        self.dataset.set_transform(self.transform)

    @abstractmethod
    def transform(batch):
        pass

    def loader(self):
        train_loader = Trainloader(
            (self.dataset.shuffle() if self.shuffle else self.dataset)["train"],
            batch_size=self.batch_size,
        )
        return train_loader


class CorpusForRNN(Corpus):
    def transform(self, batch):
        tokenized = self.tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            max_length=self.max_length * self.num_chunks,
            truncation=True,
        )
        input_ids = tokenized["input_ids"].view(-1, self.max_length)
        attention_mask = tokenized["attention_mask"].view(-1, self.max_length)
        lengths = attention_mask.sum(dim=1)
        indices = ~(lengths == 0)

        packed = pack_padded_sequence(
            input_ids[indices], lengths[indices], enforce_sorted=False, batch_first=True
        )

        to_return = {
            "data": packed.data,
            "batch_sizes": packed.batch_sizes,
            "attention_mask": tokenized["attention_mask"],
            "sorted_indices": packed.sorted_indices,
            "unsorted_indices": packed.unsorted_indices,
        }

        return to_return


class CorpusForTransformer(Corpus):
    def __init__(self, *args, masking_probability=0.15, **kw_args) -> None:
        super().__init__(*args, **kw_args)
        self.masking_probability = masking_probability
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=masking_probability
        )

    def transform(self, batch):
        tokenized = self.tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=False,
            max_length=self.max_length * self.num_chunks,
            truncation=True,
        )

        input_ids = tokenized["input_ids"].view(-1, self.max_length)
        attention_mask = tokenized["attention_mask"].view(-1, self.max_length)
        lengths = attention_mask.sum(dim=1)
        indices = ~(lengths == 0)

        input_ids, labels = self.collator.torch_mask_tokens(input_ids[indices])

        return tokenized | {
            "input_ids": input_ids,
            "attention_mask": attention_mask[indices],
            "labels": labels,
        }
