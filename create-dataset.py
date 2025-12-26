import torch
from skipgram_dataset import SkipGram
import argparse
from pathlib import Path


def create_dataset(input_path: str, output_path: str, window: int, max_training_examples: int, k: int) -> None:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    skipgram_dataset = SkipGram(device=device)
    skipgram_dataset.create(input_path, window=window, max_training_examples=max_training_examples, k=k)
    skipgram_dataset.save(output_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-file",
        type=Path,
        required=True,
        help="Path to the source text file. (.txt)",
    )

    parser.add_argument(
        "--target-file",
        type=Path,
        default=Path("skipgram_dataset.pt"),
        help="Path to the target dataset file. (.pt)",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Window size for context words.",
    )

    parser.add_argument(
        "--max-training-examples",
        type=int,
        default=100000000,
        help="Maximum number of training examples. (default: 1e8)",
    )

    parser.add_argument(
        "--negative-examples",
        type=int,
        default=5,
        help="Number of negative examples. (default: 5) Recommended: small dataset: 20, large dataset: 5",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    create_dataset(
        input_path=args.source_file,
        output_path=args.target_file,
        window=args.window,
        max_training_examples=args.max_training_examples,
        k=args.negative_examples
    )