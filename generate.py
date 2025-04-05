import argparse
import os
import torch
import pickle

from src.config import get_config
from src.model import GPT
from src.dataset import CharTokenizer, WordTokenizer # Need tokenizers for encoding/decoding
from src.utils import set_seed, mount_drive, get_device

def main(args):
    """ Main generation function. """

    # --- Setup ---
    set_seed(args.seed)
    if args.colab_drive:
        mount_drive()
    device = get_device()

    # --- Load Checkpoint and Config ---
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        config = checkpoint['config'] # Load config from checkpoint
        print("Configuration loaded from checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Fallback: Try to guess config based on filename or use args? Risky.
        print("Attempting to use default config based on args (may be incorrect)...")
        config = get_config(level=args.level, preset=args.preset)


    # Update config based on current environment/args if necessary
    config['device'] = device
    if args.max_new_tokens is not None: config['max_new_tokens'] = args.max_new_tokens
    if args.temperature is not None: config['temperature'] = args.temperature
    if args.top_k is not None: config['top_k'] = args.top_k

    print("--- Final Configuration for Generation ---")
    # Print relevant generation config
    gen_keys = ['level', 'preset', 'n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 'device']
    for k in gen_keys: print(f"{k}: {config.get(k, 'N/A')}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"top_k: {args.top_k}")
    print("-" * 20)


    # --- Tokenizer ---
    print("Loading tokenizer...")
    # We need the vocab file corresponding to the *checkpoint's* config
    vocab_path = config['vocab_path']
    if not os.path.exists(vocab_path):
         # Try constructing path relative to checkpoint if original path fails
         alt_vocab_path = os.path.join(os.path.dirname(args.checkpoint_path), os.path.basename(vocab_path))
         print(f"Original vocab path {vocab_path} not found. Trying {alt_vocab_path}")
         if os.path.exists(alt_vocab_path):
             vocab_path = alt_vocab_path
         else:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path} or {alt_vocab_path}. Cannot proceed.")


    if config['level'] == 'word':
        tokenizer = WordTokenizer(vocab_path, config['min_word_freq'])
    elif config['level'] == 'char':
        tokenizer = CharTokenizer(vocab_path)
    else:
        raise ValueError(f"Invalid level in loaded config: {config['level']}")

    if not tokenizer.load_vocab():
        raise RuntimeError(f"Failed to load vocabulary from {vocab_path}")

    # Ensure loaded vocab size matches config (important sanity check)
    if tokenizer.vocab_size != config['vocab_size']:
        print(f"Warning: Loaded vocab size ({tokenizer.vocab_size}) != config vocab size ({config['vocab_size']}). Using loaded size.")
        config['vocab_size'] = tokenizer.vocab_size

    # --- Model ---
    print("Initializing model...")
    model = GPT(config) # Initialize model with config from checkpoint
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("\nError loading state dict. Potential config mismatch between checkpoint and model definition.")
        print(e)
        print("Try ensuring the model definition in src/model.py matches the version used for training the checkpoint.")
        return # Exit if loading fails

    model.eval() # Set model to evaluation mode
    model.to(device)
    print(f"Model loaded successfully from {args.checkpoint_path}.")

    # --- Generation ---
    print("\n--- Generating Text ---")
    start_ids = tokenizer.encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0) # Add batch dim

    print(f"Prompt: '{args.prompt}' ({len(start_ids)} tokens)")
    print(f"Generating {args.max_new_tokens} new tokens...")

    with torch.no_grad():
        with torch.amp.autocast(device_type=config['device'].type, enabled=config['use_amp']):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    generated_ids = y[0].tolist() # Get list of IDs from the batch
    generated_text = tokenizer.decode(generated_ids)

    print("\n--- Generated Output ---")
    print(generated_text)
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument('checkpoint_path', type=str, help="Path to the model checkpoint (.pth file). Best practice: use 'best_model.pth'.")
    parser.add_argument('--prompt', type=str, default="Hello,", help="Starting prompt for generation.")
    parser.add_argument('--max_new_tokens', type=int, default=100, help="Number of new tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.8, help="Sampling temperature (e.g., 0.8 ~ 1.0). Lower is less random.")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k sampling threshold (e.g., 50). 0 means disabled.")
    parser.add_argument('--seed', type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument('--colab_drive', action='store_true', help="Mount Google Drive (for Colab).")

    # Optional: Specify level/preset if config loading fails (use with caution)
    parser.add_argument('--level', type=str, default='word', choices=['word', 'char'], help="Specify level if checkpoint lacks config.")
    parser.add_argument('--preset', type=str, default='tiny', choices=['nano', 'micro', 'tiny', 'small'], help="Specify preset if checkpoint lacks config.")


    args = parser.parse_args()
    if args.top_k == 0: args.top_k = None # Handle 0 as disabled

    main(args)

# Example Usage:
# python generate.py /content/drive/MyDrive/gpt2_scratch_charlevel_tiny/best_model.pth --prompt "Watson," --max_new_tokens 300 --temperature 0.75
# python generate.py /content/drive/MyDrive/gpt2_scratch_wordlevel_tiny/best_model.pth --prompt "The mystery of" --top_k 100