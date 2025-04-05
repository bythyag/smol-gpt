## GPT2 Implementation from Scratch (Training at both Word & Character Level)

The repository attemps to implements training and generation for a small GPT-2 style transformer model completely from scratch using PyTorch. It supports both **word-level** and **character-level** tokenization.

The primary goal is **educational**—to provide a clear breakdown of the model components and training process. It is designed to run on free-tier Google Colab GPUs (such as the T4).

---

### Dataset

Using the **Adventures of Sherlock Holmes** from Project Gutenberg (~581k characters):

| Level  | Preset | Vocab Size | Best Val Loss | Iter @ Best | Final Val Loss | Train Time (s) | Overfitting Trend |
|--------|--------|-------------|----------------|--------------|------------------|----------------|--------------------|
| Word   | nano   | 3291        | 4.4231         | 5000         | 4.4231           | 118            | Minimal            |
| Word   | micro  | 3291        | 4.3806         | 1500         | 5.2630           | 145            | Significant        |
| Word   | tiny   | 3291        | 4.4019         | 500          | 7.9480           | 379            | Severe             |
| Char   | nano   | 97          | 1.8109         | 5000         | 1.8109           | 118            | Minimal            |
| Char   | micro  | 97          | 1.3679         | 5000         | 1.3679           | 197            | Minimal            |
| Char   | tiny   | 97          | 1.3072         | 2000         | 1.8470           | 614            | Moderate           |

---

### Example Training Logs

Word Level - Tiny Preset (Overfitting Example)
```
Step 0: Train loss 8.2298, Val loss 8.2239
Step 500: Train loss 3.7176, Val loss 4.4019
Step 1000: Train loss 2.4839, Val loss 4.6733
Step 2000: Train loss 0.3424, Val loss 6.4148
Step 4000: Train loss 0.1414, Val loss 7.6944
Step 4999: Train loss 0.1274, Val loss 7.9480
```

Char Level - Micro Preset (Stable Learning)
```
Step 250: Train loss 2.4005, Val loss 2.4682
Step 1000: Train loss 1.8069, Val loss 1.8660
Step 2000: Train loss 1.4991, Val loss 1.5807
Step 4000: Train loss 1.2784, Val loss 1.4114
Step 4999: Train loss 1.2219, Val loss 1.3679
```

---

### Analysis

- **Overfitting**: We can see that word-level models, especially `micro` and `tiny` are overfitting quickly. However, in our experiment, the character-level models are more stable within 5k iterations.
- **Loss Interpretation**: The character-level loss values are lower due to the smaller vocabulary size but lower loss does not always mean better quality text. The ideal requirement would be to increase the dataset size and run the experiment again, this time of wikipedia type dataset. 
- **Training Time**: Larger models and character-level training are taking longer time due to longer sequence processing.
- **Model Sizes**:
  - `nano`: Stable but slow learning
  - `micro`: Best balance for character-level
  - `tiny`: Fastest learning but prone to overfitting
---

### Project Structure

```
gpt2-from-scratch/
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── model.py          # GPT model definition
│   ├── dataset.py        # Data loading and tokenization
│   ├── trainer.py        # Training loop logic
│   └── utils.py          # Utility functions
├── train.py              # Main training script
├── generate.py           # Text generation script
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md             # This file
```

---

### Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/gpt2-from-scratch.git
cd gpt2-from-scratch
```

2. **(Recommended) Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
> Make sure your `torch` installation matches your system’s CUDA version if using a GPU.

4. **(Optional: Google Colab)**  
If running in Google Colab and saving to Drive, use the `--colab_drive` flag. Ensure `drive_save_dir_base` in `src/config.py` points to a valid location.

---

### Usage

Training

**Word Level (Tiny Preset):**
```bash
python train.py --level word --preset tiny
```

**Character Level (Micro Preset):**
```bash
python train.py --level char --preset micro
```

**Colab Example:**
```bash
python train.py --level char --preset tiny --max_iters 10000 --colab_drive
```

**Override Hyperparameters:**
```bash
python train.py --level word --preset tiny --learning_rate 5e-4 --batch_size 16 --max_iters 8000
```

Checkpoints (`best_model.pth` and `latest_model.pth`) will be saved to the directory defined by `drive_save_dir`.

---

### Generation

Generate text using a trained checkpoint:

```bash
python generate.py /path/to/best_model.pth \
    --prompt "Sherlock Holmes deduced" \
    --max_new_tokens 200 \
    --temperature 0.75 \
    --top_k 40
```

**Arguments:**
- `checkpoint_path`: Path to the `.pth` model checkpoint
- `--prompt`: Initial prompt text
- `--max_new_tokens`: Number of tokens to generate
- `--temperature`: Controls randomness (lower = more focused)
- `--top_k`: Sample from top-k most likely next tokens (0 to disable)
- `--seed`: Set random seed (optional)
---
