import os
import requests
import pickle
import torch
from torch.utils.data import Dataset
from collections import Counter

# Attempt to import nltk, but make it optional for character level
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not found. Word-level tokenization will not be available.")

#------------------------------------
# Data Handling and Tokenization
#------------------------------------

class Tokenizer:
    """ Base class for tokenizers """
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        raise NotImplementedError

    def encode(self, text_string):
        raise NotImplementedError

    def decode(self, indices):
        raise NotImplementedError

    def save_vocab(self):
        if not self.stoi or not self.itos:
            print("Warning: Vocab not built yet, cannot save.")
            return
        print(f"Saving vocabulary ({self.vocab_size} items) to {self.vocab_path}...")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        with open(self.vocab_path, 'wb') as f:
            pickle.dump({'stoi': self.stoi, 'itos': self.itos}, f)
        print("Vocabulary saved.")

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            print(f"Vocabulary file not found at {self.vocab_path}.")
            return False
        try:
            with open(self.vocab_path, 'rb') as f:
                saved_vocab = pickle.load(f)
            self.stoi = saved_vocab['stoi']
            self.itos = saved_vocab['itos']
            self.vocab_size = len(self.stoi)
            print(f"Vocabulary loaded from {self.vocab_path} ({self.vocab_size} items).")
            return True
        except Exception as e:
            print(f"Error loading vocabulary from {self.vocab_path}: {e}")
            return False


class CharTokenizer(Tokenizer):
    """ Character-level tokenizer """
    def build_vocab(self, text):
        print("Building character vocabulary...")
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        print(f"Character vocabulary size: {self.vocab_size}")
        # print(f"Vocabulary: {''.join(chars)}")

    def encode(self, text_string):
        return [self.stoi.get(c, -1) for c in text_string] # Return -1 for unknown? Or handle error?

    def decode(self, indices):
        return ''.join([self.itos.get(i, '?') for i in indices]) # Use '?' for unknown indices


class WordTokenizer(Tokenizer):
    """ Word-level tokenizer using NLTK """
    def __init__(self, vocab_path, min_freq=3):
        super().__init__(vocab_path)
        self.min_freq = min_freq
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for WordTokenizer. Please install it.")
        # Download punkt if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt', quiet=True)
            print("NLTK punkt downloaded.")

    def build_vocab(self, text):
        print("Tokenizing text for word vocabulary...")
        # Lowercasing reduces vocab size
        tokens = nltk.word_tokenize(text.lower())
        print(f"Total tokens: {len(tokens)}")

        print("Building word vocabulary...")
        word_counts = Counter(tokens)
        # Filter words by minimum frequency
        filtered_word_counts = {word: count for word, count in word_counts.items() if count >= self.min_freq}

        # Add special tokens: <PAD> = 0, <UNK> = 1
        # Start actual words from index 2
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.itos = {0: '<PAD>', 1: '<UNK>'}
        # Add filtered words
        for i, word in enumerate(filtered_word_counts.keys()):
            idx = i + 2 # Start from index 2
            self.stoi[word] = idx
            self.itos[idx] = word

        self.vocab_size = len(self.stoi)
        print(f"Word vocabulary size (min_freq={self.min_freq}): {self.vocab_size}")

    def encode(self, text_string):
        words = nltk.word_tokenize(text_string.lower())
        # Use <UNK> token (index 1) for words not in vocabulary
        return [self.stoi.get(word, self.stoi['<UNK>']) for word in words]

    def decode(self, indices):
        # Use '?' for unexpected indices, skip PAD tokens (index 0) in output string
        return ' '.join([self.itos.get(i, '?') for i in indices if i != 0])


def download_data(url, save_path):
    """ Downloads text data from a URL and performs basic cleaning. """
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.content.decode('utf-8-sig') # Handle BOM

        # Basic cleaning for Project Gutenberg texts
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        start_idx = text.find(start_marker)
        if start_idx != -1: text = text[start_idx + len(start_marker):]
        end_idx = text.find(end_marker)
        if end_idx != -1: text = text[:end_idx]

        text = text.strip()
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Data downloaded and saved to {save_path}. Length: {len(text)} characters.")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return None


def prepare_data(config):
    """
    Prepares data: downloads if needed, initializes tokenizer, builds/loads vocab,
    encodes data, and returns train/val tensors and tokenizer.
    """
    data_url = config['data_url']
    data_path = config['data_path']
    level = config['level']
    vocab_path = config['vocab_path']

    # --- Get Raw Text ---
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = download_data(data_url, data_path)
        if text is None:
            raise RuntimeError("Failed to download or process data.")

    # --- Initialize Tokenizer ---
    if level == 'word':
        tokenizer = WordTokenizer(vocab_path, min_freq=config['min_word_freq'])
    elif level == 'char':
        tokenizer = CharTokenizer(vocab_path)
    else:
        raise ValueError(f"Invalid level: {level}")

    # --- Build or Load Vocabulary ---
    if not tokenizer.load_vocab():
        tokenizer.build_vocab(text)
        tokenizer.save_vocab()

    # Update config with actual vocab size
    config['vocab_size'] = tokenizer.vocab_size

    # --- Encode Data and Split ---
    print("Encoding dataset...")
    full_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded data shape: {full_data.shape}")

    n = len(full_data)
    train_data = full_data[:int(n*0.9)]
    val_data = full_data[int(n*0.9):]
    print(f"Train size: {len(train_data)} tokens, Val size: {len(val_data)} tokens")

    return train_data, val_data, tokenizer


class TextDataset(Dataset):
    """ Simple Dataset for sequential text data. """
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        # Number of possible starting positions for a sequence
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get context (x) and target (y) chunks
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y