import os

#------------------------------------
# Configuration Settings
#------------------------------------

def get_config(level='word', preset='tiny'):
    """
    Returns a configuration dictionary based on the level ('word' or 'char')
    and a size preset ('nano', 'micro', 'tiny', 'small').
    """
    assert level in ['word', 'char'], "Level must be 'word' or 'char'"
    assert preset in ['nano', 'micro', 'tiny', 'small'], "Invalid preset"

    # --- Base Configuration ---
    config = {
        'level': level,
        'preset': preset,
        # Model Hyperparameters (to be set by preset)
        'n_layer': 0,
        'n_head': 0,
        'n_embd': 0,
        # Training Hyperparameters
        'block_size': 128,       # Context length
        'batch_size': 32,        # Sequences per batch (adjust based on GPU memory)
        'max_iters': 5000,       # Total training iterations
        'eval_interval': 250,    # How often to evaluate
        'learning_rate': 3e-4,   # AdamW learning rate
        'eval_iters': 100,       # Batches for validation loss averaging
        'dropout': 0.1,          # Dropout rate
        'use_amp': True,         # Use Automatic Mixed Precision
        'grad_clip': 1.0,        # Max norm for gradient clipping (0.0 = disabled)
        # Data & Paths
        'data_url': "https://www.gutenberg.org/files/1661/1661-0.txt", # Sherlock Holmes
        'data_dir': 'data',      # Local directory to cache data
        'drive_mount_path': '/content/drive', # Default Colab Drive mount point
        'drive_save_dir_base': "/content/drive/MyDrive/gpt2_scratch", # Base dir in Drive
        'vocab_path': None,      # Set dynamically
        'model_path': None,      # Set dynamically
        'log_dir': 'logs',       # Directory for potential future logging
        'device': 'cuda',        # Set dynamically in main scripts
        'seed': 1337,
        'num_workers': 2,        # DataLoader workers (adjust based on system)
        # Level-specific settings
        'min_word_freq': 3 if level == 'word' else -1, # Word level only
        'vocab_size': -1,        # Set after data loading
    }

    # --- Model Size Presets ---
    if preset == 'nano':
        config.update({'n_layer': 3, 'n_head': 3, 'n_embd': 48})
    elif preset == 'micro':
        config.update({'n_layer': 4, 'n_head': 4, 'n_embd': 128})
    elif preset == 'tiny':
        config.update({'n_layer': 6, 'n_head': 6, 'n_embd': 384})
        # Tiny might need smaller batch size for word level on T4
        if level == 'word': config['batch_size'] = 32
        else: config['batch_size'] = 64 # Char level often fits more
    elif preset == 'small': # Original GPT-2 Small - Likely needs adjustments for T4 Free Tier
        config.update({'n_layer': 12, 'n_head': 12, 'n_embd': 768})
        config['batch_size'] = 16 # Reduce batch size significantly
        config['dropout'] = 0.1   # Original GPT-2 used 0.1

    # --- Dynamic Path Configuration ---
    level_suffix = f"{level}level_{preset}"
    config['drive_save_dir'] = f"{config['drive_save_dir_base']}_{level_suffix}"
    config['vocab_path'] = os.path.join(config['drive_save_dir'], f"{level}_vocab.pkl")
    config['model_path'] = os.path.join(config['drive_save_dir'], f"gpt2_{level_suffix}.pth")
    config['data_path'] = os.path.join(config['data_dir'], f"data_{level_suffix}.txt")

    # Derived checks/settings
    assert config['n_embd'] % config['n_head'] == 0, \
        "Embedding dimension (n_embd) must be divisible by number of heads (n_head)"
    if config['block_size'] < 64: print("Warning: block_size is quite small.")
    if level == 'char' and config['block_size'] < 128:
         print("Warning: Character models often benefit from larger block_size (e.g., 256)")

    return config

if __name__ == '__main__':
    # Example usage: Print default word and char configs
    print("--- Word Config (tiny) ---")
    word_cfg = get_config(level='word', preset='tiny')
    for k, v in word_cfg.items(): print(f"{k}: {v}")

    print("\n--- Char Config (tiny) ---")
    char_cfg = get_config(level='char', preset='tiny')
    for k, v in char_cfg.items(): print(f"{k}: {v}")