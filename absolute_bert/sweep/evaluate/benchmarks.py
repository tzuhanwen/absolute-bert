"""
這個 module 把 beir 的 IR benchmark 流程實作完整一點，只要實作
SemiSiameseBiEncodeMethod 就可以使用
"""

import os
from abc import ABC, abstractmethod
from tqdm.auto import trange

import torch

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval


try:
    import faiss
except ImportError:
    raise ImportError(
        "❌ 無法載入 `faiss` 模組。請先安裝它才能繼續使用相關 dense retrieval 功能。\n\n"
        "請根據你的環境選擇安裝方式：\n"
        "👉 CPU-only: pip install faiss-cpu\n"
        "👉 GPU-enabled: pip install faiss-gpu\n"
        "👉 conda (更穩定)：conda install -c pytorch faiss-cpu\n"
    )


def load_or_download_corpus(corpus_name="scifact", data_dir="data"):

    corpus, queries, qrels = None, None, None
    data_path = os.path.join(data_dir, corpus_name)

    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except ValueError as e:
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{corpus_name}.zip"
        )
        data_path = util.download_and_unzip(url, data_dir)
        print(data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    return corpus, queries, qrels


class SemiSiameseBiEncodeMethod(ABC):

    @abstractmethod
    def common_base(self, texts: list[str]): ...

    @abstractmethod
    def query_aggregate_fn(self, common_base_output, convert_to_tensor=True): ...

    @abstractmethod
    def corpus_aggregate_fn(self, common_base_output, convert_to_tensor=True): ...


class SemiSiameseBiEncoder:

    def __init__(self, bi_encode_method: SemiSiameseBiEncodeMethod, using_corpus_part="text"):
        self.bi_encode_method = bi_encode_method
        self.using_corpus_part = using_corpus_part

    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, self.bi_encode_method.query_aggregate_fn, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        return self.encode(
            [doc[self.using_corpus_part] for doc in corpus],
            self.bi_encode_method.corpus_aggregate_fn,
            **kwargs,
        )

    def encode(
        self,
        texts,
        aggregate_fn,
        batch_size=32,
        show_progress_bar=False,
        convert_to_tensor=True,
        **kwargs,
    ):
        results = []
        itr = (lambda *args: trange(*args, leave=False) if show_progress_bar else range)(
            0, len(texts), batch_size
        )
        for batch_start_idx in itr:
            batch_end_idx = min(batch_start_idx + batch_size, len(texts))
            batch = texts[batch_start_idx:batch_end_idx]

            result = aggregate_fn(
                self.bi_encode_method.common_base(batch), convert_to_tensor=convert_to_tensor
            )
            results.append(result)
        return torch.cat(results)


class BeirBenchmark:

    def __init__(self, corpus_name="scifact", batch_size=48):
        """
        corpus_name: scifact, trec-covid, nfcorpus
        """

        self.corpus_name = corpus_name
        self.corpus, self.queries, self.qrels = load_or_download_corpus(corpus_name=corpus_name)
        self.batch_size = batch_size

    def run(
        self,
        bi_encode_method: SemiSiameseBiEncodeMethod,
        score_fn="cos_sim",
        using_corpus_part="text",
        k_values: list[int] = [1, 3, 5, 10, 100, 1000],
        corpus_chunk_size=50000,
    ) -> NestedMetricDict:
        """
        score_fn: "dot", "cos_sim"
        """
        bi_encoder = SemiSiameseBiEncoder(bi_encode_method, using_corpus_part=using_corpus_part)
        model = DRES(bi_encoder, batch_size=self.batch_size, corpus_chunk_size=corpus_chunk_size)

        retriever = EvaluateRetrieval(
            model, score_function=score_fn, k_values=k_values
        )  # or "dot" for dot product
        results = retriever.retrieve(self.corpus, self.queries)

        #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
        metric_tuple = retriever.evaluate(self.qrels, results, retriever.k_values)

        return nest_a_metric_dict_tuple(metric_tuple)


from typing import TypeAlias

MetricDict: TypeAlias = dict[str, float]  # name@k -> value
MetricDicts: TypeAlias = tuple[MetricDict, MetricDict, MetricDict, MetricDict]
NestedMetricDict: TypeAlias = dict[str, dict[int, float]]


def nest_a_metric_dict_tuple(metric_dicts: MetricDicts) -> NestedMetricDict:

    nested_dict: dict[str, dict[int, float]] = {}  # NDCG -> "1" -> value of NDCG@1
    for metric_type, metric_dict in zip(["NDCG", "MAP", "Recall", "P"], metric_dicts):

        inner_dict: dict[int, float] = {}
        for full_name, value in metric_dict.items():
            at_value = int(full_name.strip(f"{metric_type}@"))
            inner_dict[at_value] = value

        nested_dict |= {metric_type: inner_dict}

    return nested_dict
