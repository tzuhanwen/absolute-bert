import torch
from typing import Callable, Any
from .benchmarks import SemiSiameseBiEncodeMethod

class BERTLikeSiameseSimplePoolBiEncodeMethod(SemiSiameseBiEncodeMethod):
  
  aggregate_method_dict = {
    'mean': lambda x: torch.mean(x, dim=-2),
    'sum': lambda x: torch.sum(x, dim=-2)
  }

  def __init__(
      self,
      model, 
      tokenize_fn: Callable[Any, tuple[dict[str, Any], dict[str, Any]]], 
      output_key = None, 
      common_post_fn: Callable[[torch.Tensor, dict[str, Any]], Any] = None,
      aggregate_method = 'mean',
      device = None
  ):
    """
    aggregate_method: 'mean', 'sum'
    output_key: 有些 model 的 output 是類 dict，需要再進一步取出需要的 item
    common_post_fn: input: (bert-like output embeddings, tokenize_fn output)
    tokenize_fn: output 為 (model_input_dict, byproduct_dict)，因為有些 output
      不能輸入進 model，但是 common_post_fn 要用，所以有兩個 output dict
    """

    self.model = model
    self._device = device
    self.tokenize_fn = tokenize_fn
    self.output_key = output_key
    self.common_post_fn = common_post_fn
    self.aggregate_method = aggregate_method
    
    # self.device = device

  def common_base(self, texts: list[str]):
    inputs, byproducts = self.tokenize_fn(texts)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    byproducts = {k: v.to(self.device) for k, v in byproducts.items()}

    with torch.no_grad():  
      result = self.model(**inputs)
      
      if self.output_key is not None:
        result = result[self.output_key]

      if self.common_post_fn is not None:
        result = self.common_post_fn(result, inputs|byproducts)

    return result.detach().cpu()

  def query_aggregate_fn(self, embeddings, convert_to_tensor=True):
    return self.aggregate_fn(embeddings, convert_to_tensor)

  def corpus_aggregate_fn(self, embeddings, convert_to_tensor=True):
    return self.aggregate_fn(embeddings, convert_to_tensor)

  def aggregate_fn(self, embeddings, convert_to_tensor):
    return self.aggregate_method_dict[self.aggregate_method](embeddings)

  @property
  def device(self):
    if self._device is not None:
      return self._device
    return next(self.model.parameters()).device