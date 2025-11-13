"""
這個 module 把 beir 的 IR benchmark 流程實作完整一點，只要實作
SemiSiameseBiEncodeMethod 就可以使用
"""

import logging
import os
from collections.abc import Sequence
from typing import Literal

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from absolute_bert.base_types import NestedMetricDict
from absolute_bert.bi_encoder import BiEncoder
from absolute_bert.formatter.metric import nest_a_metric_dict_tuple

logger = logging.getLogger(__name__)

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


def _load_or_download_corpus(corpus_name="scifact", data_dir="data"):

    corpus, queries, qrels = None, None, None
    data_path = os.path.join(data_dir, corpus_name)

    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except ValueError:
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{corpus_name}.zip"
        )
        data_path = util.download_and_unzip(url, data_dir)
        print(data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    return corpus, queries, qrels


class BeirBenchmark:

    def __init__(self, corpus_name="scifact"):
        """
        corpus_name: scifact, trec-covid, nfcorpus
        """

        self.corpus_name = corpus_name
        self.corpus, self.queries, self.qrels = _load_or_download_corpus(corpus_name=corpus_name)

    def run(
        self,
        bi_encoder: BiEncoder,
        batch_size: int,
        score_fn_name: Literal["dot", "cos_sim"] = "cos_sim",
        k_values: Sequence[int] = (1, 3, 5, 10, 100, 1000),
        corpus_chunk_size=50000,
    ) -> NestedMetricDict:
        logger.info(f"running beir benchmark with {bi_encoder=}, scoring method `{score_fn_name}`")

        model = DRES(bi_encoder, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)

        retriever = EvaluateRetrieval(
            model, score_function=score_fn_name, k_values=k_values
        )  # or "dot" for dot product
        results = retriever.retrieve(self.corpus, self.queries)

        #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
        metric_tuple = retriever.evaluate(self.qrels, results, retriever.k_values)

        return nest_a_metric_dict_tuple(metric_tuple)
