import torch
import model
from huggingface_hub import hf_hub_download
import pandas as pd
import torch.nn.functional as F

# Tokenizer and batch architecture

# tokenizer for context 
def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

# gets randomized training batches
def get_batch(data):
    # generates random indexes to start the batch
    ix = torch.randint(0, len(data)-block_size, (batch_size,))

# creates batch_size batches to train model 
    x = torch.stack([data[ix:ix+block_size] for i in ix])

    # creates batch_size batches, which are shifted by 1, to test model
    y = torch.stack([data[ix+1:ix+block_size+1] for i in ix])
    return x,y

# Data processing

# import corpus to one string
path = hf_hub_download(repo_id="karpathy/tiny_shakespeare",
                       filename="data/train-00000-of-00001.parquet",
                       repo_type="dataset",
                       revision="refs/pr/2")

text = pd.read_parquet(path)["text"][0]

# tokenizes characters in string data
chars = sorted(set(text))
vocab_size = len(chars)

# assigns ID labels to the tokens
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# converts string data to tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Parameter assignment
batch_size, block_size = 4, 16
d_model, n_heads, d_ff, n_layers, max_len = 32, 4, 128, 2, block_size
n_epochs = 2000

# creates model and optimizer object  
gpt_model = model(vocab_size, n_heads, d_ff, n_layers, max_len)
optimizer = torch.optim.Adam(gpt_model.parameters(), lr=1e-3)

for step in range(n_epochs):
    # fetches prediction then calculates loss 
    xb, yb = get_batch(data)
    logits = gpt_model(xb) 
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)) 

    # computes gradients and then adjusts weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Epoch: {step}, loss: {loss.item()}")




