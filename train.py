import torch
from skipgram_dataset import SkipGram
from word2vec_model import Word2Vec
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path


# Noise Contrastive Estimation Loss
def nce_loss(positive_examples, negative_examples, labels):
        nce_loss_batch = -(torch.nn.functional.logsigmoid(positive_examples.unsqueeze(-1).transpose(-2,-1) @ labels.unsqueeze(-1)).squeeze(-2,-1) + torch.sum(torch.nn.functional.logsigmoid(-positive_examples.unsqueeze(-1).transpose(-2,-1) @ negative_examples.transpose(-2,-1)).squeeze(1), dim=1))
        return torch.mean(nce_loss_batch, dim=0)

def train(path: str, output_path: str, epochs: int, embedding_dim: int=300, batch_size: int=1024, checkpoint=None, learning_rate=1e-3) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = SkipGram(device=device)
    dataset.load(path)

    word2vec = Word2Vec(voc_size=len(dataset.distribution), embedding_dim=embedding_dim).to(device)

    voc_size = len(tokenizer.vocab.keys())
    word2vec = Word2Vec(voc_size=voc_size, embedding_dim=embedding_dim).to(device)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nce_loss
    scaler = torch.amp.GradScaler(enabled=True)
    optimizer = AdamW(word2vec.parameters(), lr=learning_rate)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)

        word2vec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train_loss_list = []

    best_val_loss = float('inf')

    # Learning rate scheduler
    total_steps = epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_steps
    )

    for epoch in range(epochs):  # loop over the dataset multiple times

        # switch model to training mode
        word2vec.train()

        running_train_loss = 0.0
        running_val_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}') as pbar:
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)

                inputs = inputs.int()
                labels = labels.int()
                
                #inputs = nn.functional.one_hot(torch.tensor(inputs), num_classes=len(train.mapping.keys())).float().to(device) #Try this

                labels = word2vec.context_embedding(labels.to(device)) # We want to compare our embedding to the target or negative example
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                    outputs = word2vec.embedding(inputs)
                    negative_examples = dataset.negative_samples[torch.randint(len(dataset.negative_samples), size=(len(inputs),))].to(device)
                    negative_examples =  word2vec.context_embedding(negative_examples)
                    loss = criterion(outputs, negative_examples, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Step the scheduler
                scheduler.step()

                # print statistics
                running_train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'Loss': running_train_loss / (i + 1)})

        # Switch model to evaluation
        word2vec.eval()

        # Average losses
        avg_train_loss = running_train_loss / len(train_loader)

        train_loss_list.append(avg_train_loss)

        print(f'Epoch: {epoch + 1}, Train loss: {avg_train_loss:.3f}')

    print('Finished Training')

    torch.save({
            'model_state_dict': word2vec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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
        default=1024,
        help="Batch size for training. (default: 1024)",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to save the checkpoint. (default: None)",
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
        learning_rate=args.learning_rate
    )
