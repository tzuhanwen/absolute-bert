from collections import OrderedDict
from itertools import product

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

class Benchmarking:
  def __init__(self, tensorboard_writer, tokenizer):
    self.writer = tensorboard_writer
    self.tokenizer = tokenizer
    
    corpus_name = 'scifact'
    # corpus_name = 'trec-covid'
    # corpus_name = 'nfcorpus'
    
    self.corpus, self.queries, self.qrels = GenericDataLoader(f'data/{corpus_name}').load(split="test")
    self.corpus_text = [v['text'] for k,v in self.corpus.items()]

    
    def mean_vector(text, word_reprs):
      ids = tokenizer.encode(text, add_special_tokens=False)
      if len(ids) == 0:
          return np.zeros(word_reprs.shape[1])
      return word_reprs[ids].mean(axis=0)
    
    def idf_mean_vector(text, word_reprs):
      ids = tokenizer.encode(text, add_special_tokens=False)
      # return (vectorizer.idf_[ids] @ word_reprs[ids]) / (len(ids) + 1e-8) # 這個比較慢，可能跟 contiguous 有關
      return np.einsum('ld,l', word_reprs[ids], vectorizer.idf_[ids]) / (len(ids) + 1e-8)

    self.aggregating_method_dict = {
      'mean': mean_vector,
      'idf_mean': idf_mean_vector
    }
  
  def predict(self, word_reprs, aggregating_method, metric, part='text'):
    method = self.aggregating_method_dict[aggregating_method]
    text_vec_dict = OrderedDict({k: method(v[part], word_reprs) for k, v in self.corpus.items()})
    query_vec_dict = OrderedDict({k: method(v, word_reprs) for k, v in self.queries.items()})
    text_vecs = np.stack(list(text_vec_dict.values()))

    results = {qid: dict(zip(text_vec_dict.keys(),
                             self.score(query_vector, text_vecs, metric).tolist()
                            )
                        ) \
               for qid, query_vector in query_vec_dict.items()}

    metrics = EvaluateRetrieval.evaluate(self.qrels, results, [1, 10, 1000])
    flatten_metrics = {k: v for metric_type in metrics for k, v in metric_type.items()}
    # print(flatten_metrics)
    return flatten_metrics

  
  def predict_and_write(self,
                        word_reprs,
                        global_step,
                        aggregating_methods=['mean'],
                        metrics=['euclidean', 'cosine']
                       ):
    
    for aggregating_method, metric in product(aggregating_methods, metrics):
      indices_dict = self.predict(word_reprs, aggregating_method, metric)
      for indexName, value in indices_dict.items():
        self.writer.add_scalar(f'{metric}/{aggregating_method}/{indexName}', 
                          value,
                          global_step)
    
    
  def score(self, query_vector, text_vecs, metric):
    return (1/pairwise_distances(query_vector[None, :], text_vecs, metric=metric))[0]







    