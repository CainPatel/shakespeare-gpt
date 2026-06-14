import torch
from model import GPT   # if saved as model.py; else paste below the classes temporarily

# GPU utilization
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

# loads checkpoint
checkpoint = torch.load("checkpoints/model.pth", map_location=device)

# rebuilds model with the same architecture and hyperparameters it was trained with
# ** unpacks the config dict into the keyword arguments to match the GPT.__init__
# parameter names
model = GPT(**checkpoint["config"]).to(device)

# load trained weights into the rebuilt model
model.load_state_dict(checkpoint["model"])

# set to eval mode
model.eval()

# restore tokenizer mappings from checkpoint
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

# rebuild encode and decode
def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

# set up prompt, [] wrapping prompt makes it (1,5)
prompt = "To be"
context = torch.tensor([encode(prompt)], dtype=torch.long).to(device) 

# Generate
block_size = checkpoint["config"]["max_len"] 
out = model.generate(context, max_new_tokens=500, block_size=block_size, temperature=0.5)

# decode the result, drop batch dimension, and detensorize back into text and print
print(decode(out[0].tolist())) 


