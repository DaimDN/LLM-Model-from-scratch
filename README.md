# Transformer Language Model

A PyTorch implementation of a transformer-based language model that can be trained on any text corpus and used for text generation.

## Features

- Implements a full transformer architecture with:
  - Multi-head self-attention
  - Layer normalization
  - Positional embeddings
  - Feed-forward networks
- Character-level tokenization
- Training and evaluation loop
- Model saving/loading
- Interactive text generation

## Requirements

- Python 3.6+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install torch
```

## Usage

### Training

1. Place your training text in a file named `book.txt` in the same directory
2. Run the script:
```bash
python transformer.py
```

The script will:
- Train a new model if none exists
- Save the trained model to `transformer_model.pth`
- Enter interactive mode for text generation

### Interactive Mode

After training, you can interact with the model by typing prompts. The model will generate text based on your input.

Commands:
- Type any text prompt to generate a continuation
- Type `quit` to exit

## Configuration

Model hyperparameters can be adjusted in the `Config` class:
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimension
- `block_size`: Context window size
- `batch_size`: Training batch size
- `max_iters`: Maximum training iterations

## Implementation Details

The model implements:
- Scaled dot-product attention (with flash attention if available)
- Residual connections
- Dropout for regularization
- AdamW optimizer with weight decay
- Gradient clipping

## Example

```
Model ready. Type 'quit' to exit.

You: Once upon a time
AI: there was a little girl who lived in a small village near the forest...
```

## Notes

- For best results, train on a large text corpus (at least 1MB)
- Training on GPU is recommended for faster convergence
- The included sample text is only for testing - replace with your own data for meaningful results

## License

MIT License
