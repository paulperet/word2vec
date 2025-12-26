import torch

class Word2Vec(torch.nn.Module):

    def __init__(self, voc_size, embedding_dim) -> None:
        super(Word2Vec, self).__init__()

        self.embedding = torch.nn.Embedding(voc_size, embedding_dim) # Word Embedding | Works similarly as Linear Layer
        self.context_embedding = torch.nn.Embedding(voc_size, embedding_dim) # Map the embeddings to the outuput layer
        torch.nn.init.uniform_(self.embedding.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        torch.nn.init.uniform_(self.context_embedding.weight, -0.5/embedding_dim, 0.5/embedding_dim)