import datasets
from datasets import load_dataset

import torch

from transformers import DataCollatorForLanguageModeling

class Trainloader:
  def __init__(self, dataset, batch_size):
    self.data = dataset
    self.batch_size = batch_size

  def __len__(self):
    return len(self.data) // self.batch_size + 1

  def __iter__(self):
    current_start = 0
    for i in range(self.__len__()):
      yield self.data[current_start:current_start+self.batch_size]
      current_start += self.batch_size

# train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=collate)
# train_loader = Trainloader((dataset.shuffle() if shuffle else dataset)['train'], batch_size=batch_size)


class Load_bookcorpus:
  def __init__(self, tokenizer, cache_dir, batch_size=128, max_length=128, num_chunks=4, shuffle=False):
    self.tokenizer = tokenizer
    self.dataset = load_dataset("bookcorpus", cache_dir=cache_dir)
    self.batch_size = batch_size
    self.shuffle = shuffle

    self.max_length = max_length
    assert max_length % num_chunks == 0
    self.num_chunks = num_chunks
  
    def transform(batch):
      tokenized = tokenizer(batch['text'], return_tensors='pt', padding=True, add_special_tokens=False)
      packed = torch.nn.utils.rnn.pack_padded_sequence(
        tokenized['input_ids'], 
        tokenized['attention_mask'].sum(dim=1), 
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

    self.transform = transform
    self.dataset.set_transform(transform)

  def transformer(self, masking_probability=0.15):
    self.masking_probability = masking_probability
    self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=masking_probability)
    
    def transform(batch):
      tokenized = self.tokenizer(batch['text'],
        return_tensors='pt',
        padding='max_length',
        add_special_tokens=False,
        max_length=self.max_length*self.num_chunks,
        truncation=True)
      
      input_ids = tokenized['input_ids'].view(-1, self.max_length)
      attention_mask = tokenized['attention_mask'].view(-1, self.max_length)
      lengths = attention_mask.sum(dim=1)
      indices = ~(lengths == 0)
  
      input_ids, labels = self.collator.torch_mask_tokens(input_ids[indices])
      
      return tokenized | {'input_ids': input_ids, 'attention_mask': attention_mask[indices], 'labels': labels}

    self.transform = transform
    self.dataset.set_transform(transform)
  
  def loader(self):
    train_loader = Trainloader((self.dataset.shuffle() if self.shuffle else self.dataset)['train'], 
                               batch_size=self.batch_size)
    return train_loader

class Load_msmarco:

  def __init__(self, tokenizer, cache_dir, batch_size=128, shuffle=False):
    self.tokenizer = tokenizer
    self.dataset = load_dataset("ms_marco", 'v2.1', cache_dir=cache_dir)
    self.batch_size = batch_size
    self.shuffle = shuffle
  
    def transform(batch):
      tokenized = tokenizer(batch['query'], return_tensors='pt', padding=True, add_special_tokens=False)
      packed = torch.nn.utils.rnn.pack_padded_sequence(
        tokenized['input_ids'],
        tokenized['attention_mask'].sum(dim=1), 
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
  
    self.dataset.set_transform(transform)

  def loader(self):
    train_loader = Trainloader((self.dataset.shuffle() if self.shuffle else self.dataset)['train'], 
                               batch_size=self.batch_size)
    return train_loader

class Load_wiki:
  def __init__(self, tokenizer, cache_dir, batch_size=128, max_length=128, num_chunks=4, shuffle=False):
    self.tokenizer = tokenizer
    self.dataset = load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir)
    self.batch_size = batch_size
    self.max_length = max_length
    self.shuffle = shuffle
    assert max_length % num_chunks == 0
    self.num_chunks = num_chunks

    def transform(batch):
      tokenized = tokenizer(batch['text'],
                            return_tensors='pt',
                            padding=True,
                            add_special_tokens=False,
                            max_length=max_length*num_chunks,
                            truncation=True)
      input_ids = tokenized['input_ids'].view(-1, max_length)
      attention_mask = tokenized['attention_mask'].view(-1, max_length)
      lengths = attention_mask.sum(dim=1)
      indices = ~(lengths == 0)
      
      packed = torch.nn.utils.rnn.pack_padded_sequence(
              input_ids[indices],
              lengths[indices], 
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

    self.transform = transform
    self.dataset.set_transform(transform)

  def transformer(self, masking_probability=0.15):
    self.masking_probability = masking_probability
    self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=masking_probability)
    
    def transform(batch):
      tokenized = self.tokenizer(batch['text'],
        return_tensors='pt',
        padding=True,
        add_special_tokens=False,
        max_length=self.max_length*self.num_chunks,
        truncation=True)
      input_ids = tokenized['input_ids'].view(-1, self.max_length)
      attention_mask = tokenized['attention_mask'].view(-1, self.max_length)
      lengths = attention_mask.sum(dim=1)
      indices = ~(lengths == 0)
  
      input_ids, labels = self.collator.torch_mask_tokens(input_ids[indices])
      
      return tokenized | {'input_ids': input_ids, 'attention_mask': attention_mask[indices], 'labels': labels}

    self.transform = transform
    self.dataset.set_transform(transform)
  
  def loader(self):
    train_loader = Trainloader((self.dataset.shuffle() if self.shuffle else self.dataset)['train'], 
                               batch_size=self.batch_size)
    return train_loader