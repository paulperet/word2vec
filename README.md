# Word2Vec using Skip-Gram and Noise Contrastive Estimation

This program can be used to train word embeddings from any text file (with BERT tokenizer).

This is an implementation of the paper : 

Mikolov, Tomas & Sutskever, Ilya & Chen, Kai & Corrado, G.s & Dean, Jeffrey. (2013). 
Distributed Representations of Words and Phrases and their Compositionality. Advances in Neural Information Processing Systems.

## Installation
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
pip install -r requirements.txt
```


## Usage

### Create a dataset
```bash
python3 create-dataset.py --source-file SOURCE_FILE [--target-file TARGET_FILE] [--window WINDOW] [--max-training-examples MAX_TRAINING_EXAMPLES] [--negative-examples NEGATIVE_EXAMPLES]
```
- source file: your input text (recommended: large text file > 140mb)
- target file: name the exported dataset
- window: this is the range of context words that will be taken for each word (recommended: 5 for good syntaxic understanding, 10 for good semantic understanding)
- max training examples: maximum length of the dataset (recommended: at least 100m but you can go up to 1B and more for good quality embeddings)
- negative examples: number of randomly sampled examples (recommended: 5 for large datasets 20 for small datasets)

### Train a word2vec model
```bash
python3 train.py --dataset-path DATASET_PATH [--output-file OUTPUT_FILE] --epochs EPOCHS [--learning-rate LEARNING_RATE] [--embedding-dim EMBEDDING_DIM] [--batch-size BATCH_SIZE] [--checkpoint-path CHECKPOINT_PATH]
```
- dataset path: path of the created dataset
- output file: name the exported model
- epochs: number of training cycles (you can go very low 1-5 epochs on very large datasets)
- learning rate: how much should the embeddings be updated at each optimizer step (1e-3 to 1e-4 are good starting numbers)
- embedding dim: number of dimensions for the representation of a word (100-1000 is a recommended range, 300 is a good value)
- batch size: how many examples are used to compute the gradient (I recommend using large batches for this task 1024 or more)
- checkpoint path: resume training of a model

### Evaluate emebeddings quality
```bash
python3 evaluate.py --checkpoint-path CHECKPOINT_PATH
```
This will give you an idea of the quality of your embeddings on the wordsim353 dataset which is a human annotated dataset on words similarity. As a guideline, good embeddings usually have a 60-70% accuracy on this benchmark. The dataset is credited to:

Finkelstein, Lev, et al. "Placing search in context: The concept revisited." Proceedings of the 10th international conference on World Wide Web. ACM, 2001.

## How to use these trained embeddings in your project

You can extract the embedding layer and use bert tokenizer to use these trained embeddings in any of your projects. Each token from the tokenizer is mapped to a vector of size embedding_dim in the embeddings weights. The token directly corresponds to its index.
```python3
import torch
from transformers import AutoTokenizer

embeddings = torch.load('word2vec_model.pt')['model_state_dict']['embedding.weight']
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

word = "king"
word_id = tokenizer.vocab[word]
word_embedding = embeddings[word_id]
```

## About my implementation

### NCE Loss
For the skip-gram model, I decided to follow the negative sampling technique from the paper that results in faster training and more robust embeddings. The speed gains mainly comes from
avoiding the softmax layer and classification task and switching to a simple logistic regression task:

$$\log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^\top v_{w_I}) \right]$$

The resulting objective allows our model to bring similar words closer while moving away random sampled words. We use two embeddings matrices to train this model: one for our center words and another one for context words (positive examples) and negative examples.

### Noise Distribution
Again drawing from the paper I choose to sample my negative examples from this noise distribution, as it was outperforming any other:

$$P_n(w_i) = \frac{U(w_i)^{3/4}}{\sum_{j=1}^{n} U(w_j)^{3/4}}$$

Where $P_n(w_i)$ is the probability of word $w_i$ being selected as a negative sample, $U(w)$ is the unigram distribution (the raw frequency of the word in the corpus), $3/4$ is the empirically best power factor found in the paper and $\sum U(w_j)^{3/4}$ is the normalization factor (often denoted as $Z$) that ensures all probabilities sum to 1.

### Embedding initialization
For the embeddings initialization I choose to use a uniform distribution with a very low variance. Others have shown that low variance is the most important factor
for intialization as it avoids exploding gradients.

The dataset mainly consists of two tensors of type int16 with size MAX_TRAINING_EXAMPLES, and are loaded on the best found device: cpu, mps or cuda. Be careful as they should be able to fit in memory (As a guide: 100m examples will result in 400 MB of memory, 1B will result in 4 GB of memory used).
