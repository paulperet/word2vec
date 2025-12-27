import mmap
from torch.utils.data import Dataset
import torch
from typing import Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import os

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(data):
    return tokenizer(data["text"], truncation=False, padding=False)

# Stream sentences using memory-mapped file with capped chunk size

def stream_sentences_mmap_capped(filepath, max_chunk_size=1024*1024):
    with open(filepath, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            file_size = mm.size()
            
            while start < file_size:
                search_limit = min(start + max_chunk_size, file_size)
                dot_index = mm.find(b'.', start, search_limit)
                
                if dot_index != -1:
                    end = dot_index + 1
                    chunk = mm[start:end].decode('utf-8', errors='replace')
                    yield chunk
                    start = end
                    
                else:
                    end = search_limit
                    if end < file_size:
                        while end > start and (mm[end] & 0xC0) == 0x80:
                            end -= 1
                    chunk = mm[start:end].decode('utf-8', errors='replace')
                    yield chunk
                    start = end

class SkipGram(Dataset):
    def __init__(self, device='cpu') -> None:
        "Instantiate the dataset object"
        self.device = device

    def create(self, file_path, window=3, max_training_examples=1e7, k=20):
        "Build a dataset using skip-gram"
        self.distribution = defaultdict(int)
        self.k = k
        self.target_words = torch.ones(int(max_training_examples), dtype=torch.int16) * -1
        self.context_words = torch.ones(int(max_training_examples), dtype=torch.int16) * -1

        # Dataset creation
        dataset_index=0
        with tqdm(total=os.path.getsize(file_path), desc='Creating dataset') as pbar:
            for sentence in stream_sentences_mmap_capped(file_path):
                pbar.update(len(sentence.encode('utf-8')))
                words = tokenizer.tokenize(sentence, truncation=False)
                words = tokenizer.convert_tokens_to_ids(words)
                for index, token in enumerate(words):
                    self.distribution[token] += 1

                    left_window = 0 if index < window else index-window
                    right_window = len(words) if len(words) < index+window else window+1+index

                    for context_word in words[left_window:index] + words[index+1:right_window]:
                        if dataset_index >= max_training_examples:
                            print("Reached maximum number of training examples")
                            break
                        self.target_words[dataset_index] = int(token)
                        self.context_words[dataset_index] = int(context_word)
                        dataset_index+=1
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        self.distribution = {k: 1 - (v**(3/4) / sum([x**(3/4) for x in self.distribution.values()]))**0.5 for k, v in self.distribution.items()} 
        self.distribution = torch.tensor([self.distribution[key] for key in sorted(self.distribution.keys())])
        self.distribution = self.distribution.to(self.device)

        self.target_words = self.target_words[self.target_words != -1].to(self.device)
        self.context_words = self.context_words[self.context_words != -1].to(self.device)

        self.length = len(self.target_words)

    def save(self, filepath: str) -> None:
        "Saves the dataset to a file"
        torch.save({
            'target_words': self.target_words,
            'context_words': self.context_words,
            'distribution': self.distribution,
            'k': self.k,
            'length': self.length
        }, filepath)
    
    def load(self, filepath: str) -> None:
        "Loads the dataset from a file"
        saved_dict = torch.load(filepath, map_location=self.device)
        self.target_words = saved_dict['target_words'].to(self.device)
        self.context_words = saved_dict['context_words'].to(self.device)
        self.distribution = saved_dict['distribution'].to(self.device)



        self.k = saved_dict['k']
        self.length = saved_dict['length']

    def get_negative_examples(self, batch_size) -> torch.Tensor:
        negative_examples = self.distribution.expand(batch_size, -1).multinomial(self.k)
        return negative_examples
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.length
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.target_words[index], self.context_words[index]