import torch
from model import GPT
from huggingface_hub import hf_hub_download
import pandas as pd
import torch.nn.functional as F
import os

# Tokenizer and batch architecture

# GPT Utilization
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

# tokenizer for context 
def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

# gets randomized training batches
def get_batch(split):
    d = train_data if split == "train" else val_data
    # generates random indexes to start the batch
    ix = torch.randint(0, len(d)-block_size, (batch_size,))

# creates batch_size batches to train model 
    x = torch.stack([d[i:i+block_size] for i in ix])

    # creates batch_size batches, which are shifted by 1, to test model
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
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
data = torch.tensor(encode(text), dtype=torch.long).to(device)
n_split = int(0.9 * len(data))
train_data = data[:n_split] #first 90% of data
val_data = data[n_split:] #last 10% of data 

# Parameter assignment
batch_size, block_size = 32, 128
d_model, n_heads, d_ff, n_layers, max_len = 256, 8, 1024, 6, block_size
n_epochs = 10000
dropout = 0.1

# creates model and optimizer object  
model = GPT(vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# stops tracking gradients
@ torch.no_grad() 
def estimate_loss(split, eval_iters=50):
    # turns off dropout for clean measurement
    model.eval() 
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(split)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

for step in range(n_epochs):
    # fetches batch data and moves batch to GPU
    xbc, ybc = get_batch("train")
    xb, yb = xbc.to(device), ybc.to(device)

    # gets prediction and then calculates loss 
    logits = model(xb) 
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)) 

    # computes gradients and then adjusts weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        train_loss = estimate_loss("train")
        val_loss = estimate_loss("val")
        print(f"Epoch: {step}, Train loss: {train_loss}, Val loss: {val_loss}")

# saves model checkpoint metrics
os.makedirs("checkpoints", exist_ok=True)

torch.save({
    "model": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "max_len": max_len,
        "dropout": dropout,
    },
}, "checkpoints/model.pth")

print("Saved checkpoint to 'checkpoints/model.pth'")






