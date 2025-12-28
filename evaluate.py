import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from word2vec_model import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


def get_vector_safe(word, model):
    if word in tokenizer.vocab.keys():
        word = tokenizer.vocab[word]
        return model.embedding(torch.tensor(word).to(device)).to('cpu')
    else:
        return torch.zeros(model.embedding.embedding_dim).to('cpu')

def evaluate(checkpoint_path: str) -> None:

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get embedding dimension from the checkpoint
    embedding_dim = checkpoint['model_state_dict']['embedding.weight'].shape[1]

    voc_size = len(tokenizer.vocab.keys())
    word2vec = Word2Vec(voc_size=voc_size, embedding_dim=embedding_dim).to(device)

    word2vec.load_state_dict(checkpoint['model_state_dict'])

    wordsim353 = pd.read_csv('./data/wordsim353crowd.csv')

    wordsim353['Word Embedding 1'] = wordsim353['Word 1'].apply(lambda word: get_vector_safe(word, word2vec))
    wordsim353['Word Embedding 2'] = wordsim353['Word 2'].apply(lambda word: get_vector_safe(word, word2vec))

    wordsim353['Cosine Similarity'] = wordsim353.apply(
        lambda row: cosine_similarity([row['Word Embedding 1'].detach().numpy()], [row['Word Embedding 2'].detach().numpy()])[0][0],
        axis=1
    )

    print("\nEvaluation Results:")
    print("\n-> Spearman correlation between Word2Vec embeddings and human scores (WordSim353):")
    print(spearmanr(wordsim353['Cosine Similarity'], wordsim353['Human (Mean)']))

    embeddings = torch.load(checkpoint_path, map_location=torch.device('cpu'))['model_state_dict']['embedding.weight']
    embeddings = embeddings.numpy()
    tree = KDTree(embeddings, leaf_size=2)
    print("\nSample of words and their top-5 similarities:")
    sample_words = ['king', 'queen', 'apple', 'orange', 'computer']
    for word in sample_words:
        results = tree.query(embeddings[tokenizer.vocab[word]].reshape(1, -1), k=5)[1]
        print(f'-> {word}: {[tokenizer.convert_ids_to_tokens(int(idx)) for idx in results[0]]}')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the checkpoint file. (.pt)",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    evaluate(checkpoint_path=args.checkpoint_path)