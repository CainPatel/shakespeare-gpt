import torch
from model import GPT   # if saved as model.py; else paste below the classes temporarily

torch.manual_seed(0)
m = GPT(vocab_size=31, d_model=32, n_heads=4, d_ff=128, n_layers=2, max_len=16)

# forward
idx = torch.randint(0, 31, (2, 16))     # batch 2, seq 16
print(m(idx).shape)                      # expect (2, 16, 31)

# generate
seed = torch.randint(0, 31, (1, 4))      # batch 1, seq 4
out = m.generate(seed, max_new_tokens=20, block_size=16)
print(out.shape)                         # expect (1, 24)  -- 4 seed + 20 new