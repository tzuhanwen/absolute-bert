from . import bi_encoder_methods
from .benchmarks import BeirBenchmark, CommonBaseBiEncodeMethod

BI_ENCODER_METHODS = {
    "bert_like_symmetric": bi_encoder_methods.BERTLikeSymmetricBiEncodeMethod
}

