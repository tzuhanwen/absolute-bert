import datasets
from datasets import load_dataset

import torch

def load_bookcorpus(tokenizer, cache_dir):
  dataset = load_dataset("bookcorpus", cache_dir=cache_dir)

  def transform(batch):
    tokenized = tokenizer(batch['text'], return_tensors='pt', padding=True, add_special_tokens=False)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
      tokenized['input_ids'], tokenized['attention_mask'].sum(dim=1), 
      enforce_sorted=False,
      batch_first=True
    )
    
    to_return = {
      'data': packed.data,
      'batch_sizes': packed.batch_sizes,
      'attention_mask': tokenized['attention_mask'],
      'sorted_indices': packed.sorted_indices,
      'unsorted_indices': packed.unsorted_indices
    }
    
    return to_return
  
  dataset.set_transform(transform)

  return dataset

def load_msmarco(tokenizer, cache_dir):
  dataset = load_dataset("ms_marco", 'v2.1', cache_dir=cache_dir)

  def transform(batch):
    tokenized = tokenizer(batch['query'], return_tensors='pt', padding=True, add_special_tokens=False)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
      tokenized['input_ids'], tokenized['attention_mask'].sum(dim=1), 
      enforce_sorted=False,
      batch_first=True
    )
    
    to_return = {
      'data': packed.data,
      'batch_sizes': packed.batch_sizes,
      'attention_mask': tokenized['attention_mask'],
      'sorted_indices': packed.sorted_indices,
      'unsorted_indices': packed.unsorted_indices
    }
    
    return to_return
  
  dataset.set_transform(transform)

  return dataset