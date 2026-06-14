import torch
from model import GPT   # if saved as model.py; else paste below the classes temporarily
import argparse

def load_model(device):
    # loads checkpoint
    checkpoint = torch.load("checkpoints/model.pth", map_location=device)

    # rebuilds model with the same architecture and hyperparameters it was trained with
    model = GPT(**checkpoint["config"]).to(device)

    # load trained weights into the rebuilt model
    model.load_state_dict(checkpoint["model"])

    # set to eval mode
    model.eval()

    # restore tokenizer mappings from checkpoint and returns functions
    return model, checkpoint["stoi"], checkpoint["itos"]

def main():
    # define command line arguments
    parser = argparse.ArgumentParser(description="Generate Shakespearean text")
    parser.add_argument("--prompt", type=str, default="To be", help="Starting text for generation")
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help="Sampling temperature. Lower = more conservative, higher = more creative")
    parser.add_argument("--tokens", type=int, default=500, help="Number of characters to generate")
    args = parser.parse_args()

    # GPU utilization
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, stoi, itos = load_model(device)

    # rebuild encode and decode
    def encode(s): return [stoi[c] for c in s]
    def decode(ids): return "".join(itos[i] for i in ids)

    # set up context, [] wrapping prompt makes it (1,5), and reads block size from model
    context = torch.tensor([encode(args.prompt)], dtype=torch.long).to(device) 
    block_size = model.pos_emb.num_embeddings

    # generates output
    out = model.generate(context, max_new_tokens=args.tokens, block_size=block_size, temperature=args.temperature)

    # decode the result, drop batch dimension, and detensorize back into text and print
    print(decode(out[0].tolist())) 

if __name__ == "__main__":
    main()


