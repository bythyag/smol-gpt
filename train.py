import argparse
import os
import torch

from src.config import get_config
from src.model import GPT
from src.dataset import prepare_data, TextDataset
from src.trainer import Trainer
from src.utils import set_seed, mount_drive, get_device

def main(args):
    """ Main training function. """

    # --- Setup ---
    set_seed(args.seed)
    if args.colab_drive:
        mount_drive() # Mount drive if running in Colab and requested

    device = get_device()

    # --- Configuration ---
    # Load config, potentially overriding some values with args
    config = get_config(level=args.level, preset=args.preset)
    config['device'] = device
    config['seed'] = args.seed
    if args.max_iters is not None: config['max_iters'] = args.max_iters
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.learning_rate is not None: config['learning_rate'] = args.learning_rate
    if args.eval_interval is not None: config['eval_interval'] = args.eval_interval
    if args.block_size is not None: config['block_size'] = args.block_size
    if args.n_layer is not None: config['n_layer'] = args.n_layer
    if args.n_head is not None: config['n_head'] = args.n_head
    if args.n_embd is not None: config['n_embd'] = args.n_embd

    print("--- Configuration ---")
    for k, v in config.items(): print(f"{k}: {v}")
    print("-" * 20)

    # --- Data ---
    print("Preparing data...")
    train_data, val_data, tokenizer = prepare_data(config)
    # Config vocab_size is updated inside prepare_data
    print(f"Vocabulary size: {config['vocab_size']}")

    train_dataset = TextDataset(train_data, config['block_size'])
    val_dataset = TextDataset(val_data, config['block_size'])
    print("Data preparation complete.")

    # --- Model ---
    print("Initializing model...")
    model = GPT(config)
    model.to(device)
    # Optional: Compile model for speedup (PyTorch 2.0+)
    # model = torch.compile(model)
    print("Model initialized.")

    # --- Trainer ---
    print("Initializing trainer...")
    trainer = Trainer(model, config, train_dataset, val_dataset)
    print("Trainer initialized.")

    # --- Train ---
    print("Starting training...")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model from scratch.")
    parser.add_argument('--level', type=str, default='word', choices=['word', 'char'],
                        help="Tokenization level ('word' or 'char').")
    parser.add_argument('--preset', type=str, default='tiny', choices=['nano', 'micro', 'tiny', 'small'],
                        help="Model size preset.")
    parser.add_argument('--seed', type=int, default=1337, help="Random seed.")
    parser.add_argument('--colab_drive', action='store_true', help="Mount Google Drive (for Colab).")

    # Allow overriding key config parameters via command line
    parser.add_argument('--max_iters', type=int, default=None, help="Override max training iterations.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size.")
    parser.add_argument('--learning_rate', type=float, default=None, help="Override learning rate.")
    parser.add_argument('--eval_interval', type=int, default=None, help="Override evaluation interval.")
    parser.add_argument('--block_size', type=int, default=None, help="Override block size (context length).")
    parser.add_argument('--n_layer', type=int, default=None, help="Override number of layers.")
    parser.add_argument('--n_head', type=int, default=None, help="Override number of heads.")
    parser.add_argument('--n_embd', type=int, default=None, help="Override embedding dimension.")

    args = parser.parse_args()
    main(args)

# Example Usage:
# python train.py --level char --preset micro --max_iters 10000 --colab_drive
# python train.py --level word --preset tiny --learning_rate 5e-4