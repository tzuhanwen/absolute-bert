from abc import ABC, abstractmethod
import datasets
from datasets import load_dataset
import torch
from transformers import DataCollatorForLanguageModeling


# https://github.com/huggingface/datasets/blob/main/src/datasets/load.py
datasets_args = {
    "bookcorpus": {
        "path": "bookcorpus",
        "extracting_part": "text"
    },
    "msmarco": {
        "path": "ms_marco",
        "name": "v2.1",
        "extracting_part": "query"
    },
    "wiki": {
        "path": "wikipedia",
        "name": "20220301.en",
        "extracting_part": "text"
    }
}



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

class Corpus(ABC):
    def __init__(self, corpus_name, tokenizer, cache_dir, batch_size=128, max_length=128, num_chunks=4, shuffle=False):
        "Extract the first <max_length>*<num_chunks> tokens of each text in the corpus, and split into <num_chunks> data."
    
        self.corpus_name = corpus_name
        self.tokenizer = tokenizer
    
        dataset_args = datasets_args[corpus_name]
        self.dataset = load_dataset(dataset_args["path"], 
            dataset_args["name"] if "name" in dataset_args else None,
            cache_dir=cache_dir,
            trust_remote_code=True)
      
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
        train_loader = Trainloader((self.dataset.shuffle() if self.shuffle else self.dataset)['train'], 
            batch_size=self.batch_size)
        return train_loader



class Corpus_for_RNN(Corpus):
    def transform(self, batch):
        tokenized = tokenizer(batch['text'],
            return_tensors='pt',
            padding=True,
            add_special_tokens=False,
            max_length=self.max_length*self.num_chunks,
            truncation=True)
        input_ids = tokenized['input_ids'].view(-1, self.max_length)
        attention_mask = tokenized['attention_mask'].view(-1, self.max_length)
        lengths = attention_mask.sum(dim=1)
        indices = ~(lengths == 0)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_ids[indices],
            lengths[indices], 
            enforce_sorted=False,
            batch_first=True)
        
        to_return = {
            'data': packed.data,
            'batch_sizes': packed.batch_sizes,
            'attention_mask': tokenized['attention_mask'],
            'sorted_indices': packed.sorted_indices,
            'unsorted_indices': packed.unsorted_indices
        }
        
        return to_return


class Corpus_for_transformer(Corpus):
    def __init__(self, *args, masking_probability=0.15, **kw_args):
        super().__init__(*args, **kw_args)
        self.masking_probability = masking_probability
        self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=masking_probability)
    
    def transform(self, batch):
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
        