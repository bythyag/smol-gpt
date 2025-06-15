## gpt2 implementation and training (word & character level tokenization)

the project attemps to implement and train a small gpt-2 style transformer model using pytorch. it also explores both **word-level** and **character-level** tokenization across different architecture and parameter settings to study loss and training time.

the primary goal of the proejct is **educational**—to provide a clear breakdown of the model components and training process. it is designed to run on free-tier google colab gpus (such as the t4).

## dataset

using the **adventures of sherlock holmes** from project gutenberg (~581k characters):

| level  | preset | vocab size | best val loss | iter @ best | final val loss | train time (s) | overfitting trend |
|--------|--------|-------------|----------------|--------------|------------------|----------------|--------------------|
| word   | nano   | 3291        | 4.4231         | 5000         | 4.4231           | 118            | minimal            |
| word   | micro  | 3291        | 4.3806         | 1500         | 5.2630           | 145            | significant        |
| word   | tiny   | 3291        | 4.4019         | 500          | 7.9480           | 379            | severe             |
| char   | nano   | 97          | 1.8109         | 5000         | 1.8109           | 118            | minimal            |
| char   | micro  | 97          | 1.3679         | 5000         | 1.3679           | 197            | minimal            |
| char   | tiny   | 97          | 1.3072         | 2000         | 1.8470           | 614            | moderate           |

## notes

- **overfitting**: we can see that word-level models, especially `micro` and `tiny` are overfitting quickly. however, in our experiment, the character-level models are more stable within 5k iterations.
- **loss interpretation**: the character-level loss values are lower due to the smaller vocabulary size but lower loss does not always mean better quality.
- **training time**: larger models and character-level training are taking longer time due to longer sequence processing.
- **model sizes**:
  - `nano`: stable but slow learning
  - `micro`: best balance for character-level
  - `tiny`: fastest learning but prone to overfitting

## llm usage

his is an exporation/educational project whose codes were generated using ai tools. used generative ai tools to write gpt2 model on pytorch, implement tokenization, and training script. 