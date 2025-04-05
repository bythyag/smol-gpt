# smolgpt (Word & Character Level)

This repository implements training and generation for a small GPT-2 style transformer model completely from scratch using PyTorch. It supports both word-level and character-level tokenization.

The primary goal is educational—providing a clear breakdown of the components and training process. It's designed to be runnable on free-tier Google Colab GPUs (like T4).

---

## Features

- **GPT-2 Architecture**: Implements core components like multi-head causal self-attention, MLP blocks, Layer Normalization, positional embeddings, and weight tying from scratch.
- **Word & Character Levels**: Supports both word-level (using NLTK) and character-level tokenization.
- **Configurable**: Easily switch between tokenization levels and model size presets (`nano`, `micro`, `tiny`, `small`). Key hyperparameters are configurable via command-line arguments or the `src/config.py` file.
- **Training**: Includes a training script (`train.py`) with:
  - AdamW optimizer
  - Automatic Mixed Precision (AMP) for faster training and lower memory usage
  - Gradient clipping
  - Validation loss calculation
  - Checkpoint saving (best and latest models) to Google Drive (if using Colab) or locally
  - Resuming from checkpoints
- **Generation**: Includes a generation script (`generate.py`) with:
  - Loading trained checkpoints
  - Sampling options: temperature and top-k
- **Structured Code**: Organized into modules for clarity and reusability (`src/`)
- **Colab Friendly**: Includes utilities for mounting Google Drive

---

## Project Structure

```
gpt2-from-scratch/
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── model.py          # GPT model definition
│   ├── dataset.py        # Data loading, tokenization, vocab
│   ├── trainer.py        # Training loop logic
│   └── utils.py          # Utility functions
├── train.py              # Main training script
├── generate.py           # Main generation script
├── requirements.txt      # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

---

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/gpt2-from-scratch.git  # Replace your-username
cd gpt2-from-scratch
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
# venv\Scripts\activate   # For Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> 💡 Make sure your `torch` installation matches your system's CUDA version if using a GPU.

4. **(Optional - Colab/Drive)**  
   If running on Google Colab and want to save checkpoints/vocab to Google Drive, use the `--colab_drive` flag.  
   Make sure `drive_save_dir_base` in `src/config.py` points to a valid location in your Drive.

---

## Usage

### Training

Train a new model using `train.py`.

#### Basic Usage (Word Level, Tiny Preset):

```bash
python train.py --level word --preset tiny
```

#### Basic Usage (Character Level, Micro Preset):

```bash
python train.py --level char --preset micro
```

#### Colab Example (Tiny Char Model, Saving to Drive):

```bash
python train.py --level char --preset tiny --max_iters 10000 --colab_drive
```

#### Overriding Parameters:

```bash
python train.py --level word --preset tiny --learning_rate 5e-4 --batch_size 16 --max_iters 8000
```

Training logs and loss information will be printed to the console.

Checkpoints (`best_model.pth` and `latest_model.pth`) will be saved in the directory specified by `drive_save_dir` in the config (e.g., `/content/drive/MyDrive/gpt2_scratch_wordlevel_tiny/`).

---

### Generation

Use the `generate.py` script to generate text using a trained checkpoint.

#### Example:

```bash
python generate.py /path/to/your/drive/MyDrive/gpt2_scratch_charlevel_tiny/best_model.pth \
  --prompt "Sherlock Holmes deduced" \
  --max_new_tokens 200 \
  --temperature 0.75 \
  --top_k 40
```

#### Arguments:

- `checkpoint_path`: Path to the `.pth` model checkpoint.
- `--prompt`: The initial text to start generation from.
- `--max_new_tokens`: How many new tokens (words or characters) to generate.
- `--temperature`: Controls randomness. Lower values (e.g., 0.7) = more focused output.
- `--top_k`: Considers only the top-k most likely next tokens. Set to 0 to disable.
- `--seed`: Random seed for reproducibility.

---

## Notes

- **GPU Memory**: Training even 'tiny' or 'small' models can be memory-intensive. If you get OOM errors, reduce `batch_size`, `block_size`, or use a smaller model preset. AMP (`use_amp=True`) helps a lot with memory.
- **Word Tokenization**: Uses NLTK with lowercase filtering and frequency-based vocab cutoff. Adds `<PAD>` and `<UNK>` tokens. Simpler than BPE (used in real GPT-2), but easier to understand.
- **Character Tokenization**: Robust and simple. Handles any input text but typically needs longer training and/or bigger models to capture dependencies.
- **Training Time**: Training can take several hours depending on dataset size, model, and `max_iters`. Be patient!

---
