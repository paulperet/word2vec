# Word2Vec using Skip-Gram and Noise Contrastive Estimation

This program can be used to train a word embedding model from any text file (with BERT tokenizer). 
This is an implementation of: Mikolov, Tomas & Sutskever, Ilya & Chen, Kai & Corrado, G.s & Dean, Jeffrey. (2013). 
Distributed Representations of Words and Phrases and their Compositionality. Advances in Neural Information Processing Systems. 26. 

# Installation
Project used Python 3.12.9 but other versions may be suitable

Clone the project:
```bash
git clone https://github.com/paulperet/word2vec
```

Go to the project's folder
```bash
cd word2vec
```

Create a virtual environment
```bash
python -m venv .venv
```

Activate the environment
```bash
source .venv/bin/activate
```

Install the dependancies
```bash
pip install -r requirements
```


# Usage

Create a dataset
```bash
python3 create-dataset.py --source-file SOURCE_FILE [--target-file TARGET_FILE] [--window WINDOW] [--max-training-examples MAX_TRAINING_EXAMPLES] [--negative-examples NEGATIVE_EXAMPLES]
```

Train a word2vec model
```bash
python3 train.py --dataset-path DATASET_PATH [--output-file OUTPUT_FILE] --epochs EPOCHS [--learning-rate LEARNING_RATE] [--embedding-dim EMBEDDING_DIM] [--batch-size BATCH_SIZE] [--checkpoint-path CHECKPOINT_PATH]
```

# How to use these trained embeddings in your project

You can extract the embedding layer and use bert tokenizer to use these trained embeddings in any of your projects. Each token from the tokenizer is mapped to a vector of size embedding_dim in the embeddings weights. The token directly corresponds to its index.
```python3
import torch
from transformers import AutoTokenizer

embeddings = torch.load('word2vec_model.pt')['model_state_dict']['embedding.weight']
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```
