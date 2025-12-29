import torch
from word2vec_model import Word2Vec
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Noise Contrastive Estimation Loss
def nce_loss(positive_examples, negative_examples, labels):
    "Computes the NCE loss for a batch of positive and negative examples."
    nce_loss_batch = -(torch.nn.functional.logsigmoid(torch.bmm(positive_examples.unsqueeze(-1).transpose(-2,-1), labels.unsqueeze(-1))).squeeze(-2,-1) + torch.sum(torch.nn.functional.logsigmoid(-torch.bmm(positive_examples.unsqueeze(-1).transpose(-2,-1), negative_examples.transpose(-2,-1))).squeeze(1), dim=1))
    return torch.mean(nce_loss_batch, dim=0)

def get_sampler(path: str, tokenizer: Tokenizer):
    "Returns a unigram distribution raised to the 3/4 power for negative sampling."
    distribution = Counter()
    with open(path, 'r') as file:
        while True:
            chunk = file.read(1024*1024)
            words = chunk.partition(' ')[-1][::-1].partition(' ')[-1][::-1] # Trim partial words
            tokens = tokenizer.encode(words).ids
            distribution.update(tokens)
            break
    sampler = torch.ones(len(tokenizer.get_vocab()))
    for word_id, count in distribution.items():
        sampler[word_id] = count
    final_sampler = sampler**0.75 / (sampler**0.75).sum()
    return final_sampler

def get_examples(tokens: torch.Tensor, window: int) -> torch.Tensor:
    "Returns a 2D tensor of (input, label) pairs for skip-gram model."
    n_tokens = tokens.size(0)

    # Get center words indices and context words indices
    center_indices = torch.arange(n_tokens).view(-1, 1)
    offsets = torch.cat([torch.arange(-window, 0), torch.arange(1, window + 1)])

    # Use broadcasting to get all context indices
    context_indices = center_indices + offsets

    # Use mask to filter out-of-bounds indices
    mask = (context_indices >= 0) & (context_indices < n_tokens)

    rows = center_indices.repeat(1, 2 * window)[mask]
    cols = context_indices[mask]

    examples = torch.stack((tokens[rows], tokens[cols]), dim=1)
    return examples

def train(path: str, output_path: str, epochs: int, embedding_dim: int=300, batch_size: int=10000, checkpoint=None, learning_rate=1e-3, num_workers=4, k=5, window=5) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=100000, min_frequency=5)
    tokenizer.train(files=[os.fspath(path)], trainer=trainer)
    tokenizer.save(os.getcwd() + '/tokenizer.json')

    voc_size = len(tokenizer.get_vocab().keys())
    word2vec = Word2Vec(voc_size=voc_size, embedding_dim=embedding_dim).to(device)
    
    criterion = nce_loss
    if device == "cuda":
        use_amp = True
    else:
        use_amp = False

    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer = AdamW(word2vec.parameters(), lr=learning_rate)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)

        word2vec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    mean_length_word = 6
    number_of_words = batch_size * mean_length_word
    chunk_size = number_of_words//(4*window)

    # Learning rate scheduler

    number_of_batches = os.path.getsize(path) // chunk_size
    gradient_accumulation_steps = number_of_batches // 500 if number_of_batches // 500 > 0 else 1

    steps_per_epoch = (number_of_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_steps = steps_per_epoch * epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-2, 
        total_steps=total_steps
    )

    sampler = get_sampler(path, tokenizer)

    # Pre-sample negative examples
    sampled_negative_examples = torch.multinomial(sampler, batch_size * k * 10, replacement=True).to(device)

    for epoch in range(epochs):  # loop over the dataset multiple times

        # switch model to training mode
        word2vec.train()

        running_train_loss = 0.0
        i = 0
        with tqdm(total=number_of_batches, desc=f'Epoch {epoch + 1}') as pbar:
            with open(path, 'r') as file:
                for _ in range(number_of_batches):
                    chunk = file.read(chunk_size)
                    ## Process chunk here

                    words = chunk.partition(' ')[-1].rpartition(' ')[0] # Trim partial words
                    tokens = torch.tensor(tokenizer.encode(words).ids, dtype=torch.int64) # Encode the chunk
                
                    examples = get_examples(tokens, window) # Get (input, label) pairs
                    
                    inputs_id = examples[:, 0].to(device)
                    labels_id = examples[:, 1].to(device)

                    random_indices = torch.randint(0, len(sampled_negative_examples), (examples.shape[0]*k,))
                    negative_examples_id = sampled_negative_examples[random_indices].view(examples.shape[0], k)

                    # forward + backward + optimize
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

                        # Get embeddings
                        positive_examples = word2vec.embedding(inputs_id)
                        labels = word2vec.context_embedding(labels_id) # We want to compare our embedding to the target or negative example
                        negative_examples = word2vec.context_embedding(negative_examples_id)

                        # Compute loss
                        loss = criterion(positive_examples, negative_examples, labels)
                        loss = loss / gradient_accumulation_steps
                        
                    scaler.scale(loss).backward()
    
                    running_train_loss += loss.item()

                    # Gradient accumulation
                    if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == number_of_batches:
                        scaler.step(optimizer)
                        scaler.update()

                        # Step the scheduler
                        scheduler.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                    # print statistics
                    pbar.update(1)
                    pbar.set_postfix({'Loss': running_train_loss / ((i + 1) / gradient_accumulation_steps)})

                    i+=1

            # Average losses
            avg_train_loss = running_train_loss / (number_of_batches / gradient_accumulation_steps)

            print(f'Epoch: {epoch + 1}, Train loss: {avg_train_loss:.3f}')

    print('Finished Training')

    torch.save({
            'model_state_dict': word2vec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
            }, output_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset file. (.pt)",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("word2vec_model.pt"),
        help="Path to the target model file. (.pt)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer. (default: 1e-3)",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=300,
        help="Dimension of the embedding vectors.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for training. (default: 10000)",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to save the checkpoint. (default: None)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading. (default: 4)",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    train(
        path=args.dataset_path,
        output_path=args.output_file,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint_path,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers
    )
