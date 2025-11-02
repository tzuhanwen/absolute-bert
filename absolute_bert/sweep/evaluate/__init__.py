from . import bi_encoder_methods
from .benchmarks import BeirBenchmark, SemiSiameseBiEncodeMethod

BI_ENCODER_METHODS = {
    "bert_like_siamese_simple_pool": bi_encoder_methods.BERTLikeSiameseSimplePoolBiEncodeMethod
}

