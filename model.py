# Shakespear GPT

import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer Architecture

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads 
        
        # initializes trackable weight matrices (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, n, d_model = x.shape
        # projects inputs to create informationally encoded matrix (n, d_model)
        K = self.W_k(x) 
        Q = self.W_q(x)
        V = self.W_v(x)

        # reshapes vector to allow for parallel computation of multihead attention (batch, n, n_heads, d_k)
        K = K.view(batch, n, self.n_heads, self.d_k) 
        Q = Q.view(batch, n, self.n_heads, self.d_k)
        V = V.view(batch, n, self.n_heads, self.d_k)

        # reshapes vector to allow d_k to be contracted during matmul (batch, n_heads, n, d_k)
        K = K.transpose(1,2)
        Q = Q.transpose(1,2)
        V = V.transpose(1,2)

        # Calculates key-query compatibility score (batch, n_heads, n, n)
        score = Q @ K.transpose(-1,-2) 

        # standardizes score
        st_score = score / (self.d_k**0.5) 

        # masks future tokens 
        mask = torch.tril(torch.ones(n,n, device=x.device))
        masked_score = st_score.masked_fill(mask == 0, float('-inf'))

        # converts scores to weights 
        weights = torch.softmax(masked_score, dim=-1)

        # calculates attention of each token (batch, n_heads, n, d_k)
        causal_vector = weights @ V

        # reshapes vector then concatenates to form multihead attention
        causal_vector = causal_vector.transpose(1,2)
        multihead_vector = causal_vector.reshape(batch, n, d_model)

        # projects multihead vector to weights to get multihead attention vector
        multihead_attention = self.W_o(multihead_vector)
        return multihead_attention
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff) 
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # expands dimensions, applies nonlinear transformation, contracts to distils features
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # output = residual + norm layer
        res1 = x + self.attn(self.ln1(x))
        output = res1 + self.ff(self.ln2(res1))
        return output

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model) # Token ID 
        self.pos_emb = nn.Embedding(max_len, d_model) # Position Vector is learned
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # creates a token vector based on ID (batch, n, d_model) and position (n, d_model)
        batch, n = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(n, device=idx.device))
        x = tok + pos

        # runs transformer blocks then applies a final normalization (n, d_model) 
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # turned vector into logits (n, vocab_size)
        logits = self.head(x)
        return logits 
    
    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            # crops context window to last blocksize number of tokens (ID, n)
            context = idx[:, -block_size:]

            # projects Token ID to learned logits of importance at each token position (batch, n, vocab_size)
            logits = self(context) 

            # isolates the logits at final token
            logits = logits[:,-1,:]

            # converts the logits to prob distribution for next character id then appends sample
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx