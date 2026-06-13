import torch

# Tokenizer, batch architecture

block_size, batch_size = 1,1 

# import data to train the transformer  
text = ""

# tokenizes characters in string data
chars = sorted(set(text))
vocab_size = len(chars)

# assigns ID labels to the tokens
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

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