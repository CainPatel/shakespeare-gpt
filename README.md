# Shakespeare GPT

A decoder-only GPT built entirely from scratch in PyTorch — every component implemented by hand (no `nn.Transformer`) — trained to generate Shakespearean text.

## What this is

This is a character-level GPT language model. I implemented every part of the transformer architecture from first principles to understand the internals: multi-head causal self-attention, positional embeddings, the feed-forward network, residual connections, layer normalization, and dropout. The only PyTorch building blocks used are `nn.Linear`, `nn.Embedding`, and `nn.LayerNorm` — the attention mechanism, masking, and multi-head logic are all hand-written.

The model is trained on the [tiny-shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) corpus (~1MB of Shakespeare's plays) and generates novel text in a Shakespearean style.

## Sample output

Generated from the prompt `"To be"` (temperature 0.8):

```
To be patience than the less blood with our crown.
KING RICHARD III:
Then shall dam, of the stars what you sleep-change now,
Which you could be patience keep their worships,
It be guilty to tell the people's since
Music ere your crowns with his heart. Tut, sir,
The plague of Juliet? What thou art a little come?
CATESBY:
Ay, his son, why, sir, hast thou not came my lord will come
To hither by our live, to speak it about
Your own beautiful cousin?
```

The model learns real words, correct spelling, play structure (character names, dialogue), and Shakespearean phrasing — though as a small character-level model it captures *style* far better than *meaning*.

## Temperature

Generation samples from the model's predicted distribution, reshaped by a temperature parameter:

| Temperature | Behavior |
|-------------|----------|
| **0.5** | Coherent and grammatical, but repetitive and conservative |
| **0.8** | The sweet spot — coherent yet varied |
| **1.2** | Creative and surprising, but more invented words and errors |

Lower temperature sharpens the distribution toward high-probability tokens; higher temperature flattens it, giving rare tokens more chance.

## Architecture

A decoder-only transformer:

- Token + learned positional embeddings
- 6 transformer blocks, each with:
  - Multi-head causal self-attention (8 heads)
  - Position-wise feed-forward network (GELU)
  - Pre-norm residual connections
  - Dropout
- Final layer norm + linear projection to vocabulary logits

Default configuration: `d_model=256`, `n_layers=6`, `n_heads=8`, `d_ff=1024`, `block_size=128`, `dropout=0.1`. Trained with Adam, cross-entropy loss, and a 90/10 train/validation split. Supports Apple Silicon GPU (MPS) and CPU.

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (downloads the dataset automatically)
python train.py

# Generate
python generate.py --prompt "To be" --temperature 0.8 --tokens 500
```

## Notes

I built this to understand transformers at the implementation level — the kind of understanding you only get from writing the attention math, the causal mask, and the multi-head reshape yourself, then debugging them. Loss reached ~1.19 on the validation set.

## License

MIT © Cain Patel