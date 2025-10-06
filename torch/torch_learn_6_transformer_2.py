import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Config / hyperparams
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Use a larger seq_len for "longer" sequences — adjust to your memory / GPU:
seq_len = 64          # context window (increase for longer context)
batch_size = 16
d_model = 128         # embedding dimension (must be divisible by nhead)
nhead = 8             # number of attention heads -> d_model % nhead == 0
num_layers = 4
lr = 3e-4
epochs = 2
iters_per_epoch = 200
max_generate = 80     # how many tokens to generate at inference time
grad_clip = 1.0

# --------------------------
# Tiny tokenizer (word + punctuation)
# --------------------------
def tokenize(text):
    # returns list of tokens, lowercased: words and punctuation are separate tokens
    return re.findall(r"\w+|[^\s\w]", text.lower())

# Example small corpus — replace with your dataset (list of lines or one big string)
corpus = (
    "Alice was beginning to get very tired of sitting by her sister on the bank. "
    "She had nothing to do. Once or twice she had peeped into the book her sister was reading, "
    "but it had no pictures or conversations in it."
)

tokens = tokenize(corpus)
# Build vocab
vocab = sorted(set(tokens))
stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# Create a long training stream by repeating the corpus (toy demo). Replace with your dataset load for real tasks.
repeats = 100
data = [stoi[t] for t in tokens] * repeats
data_len = len(data)
print("Data length (tokens):", data_len)

# --------------------------
# Dataset batching utilities
# --------------------------
def get_random_batch(batch_size, seq_len):
    xs = []
    ys = []
    for _ in range(batch_size):
        start = random.randint(0, data_len - seq_len - 2)
        x = torch.tensor(data[start : start + seq_len], dtype=torch.long)
        y = torch.tensor(data[start + 1 : start + seq_len + 1], dtype=torch.long)
        xs.append(x)
        ys.append(y)
    # stacks: (batch, seq_len) -> transpose to (seq_len, batch)
    xs = torch.stack(xs).t().contiguous().to(device)
    ys = torch.stack(ys).t().contiguous().to(device)
    return xs, ys

# --------------------------
# Model: token + pos emb + TransformerEncoder with causal mask
# --------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, nhead, num_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        # x: (seq_len, batch)
        seq_len, batch = x.shape
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        h = self.tok_emb(x) + self.pos_emb(positions).unsqueeze(1)  # (seq_len, batch, d_model)
        # causal mask (square subsequent)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(h, mask=mask)  # (seq_len, batch, d_model)
        out = self.ln_f(out)
        logits = self.head(out)  # (seq_len, batch, vocab)
        return logits

model = MiniGPT(vocab_size=vocab_size, max_len=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# --------------------------
# Sampling utilities (greedy or sampling with temperature & top-k)
# --------------------------
def top_k_logits(logits, k):
    # logits: (vocab,)
    if k == 0:
        return logits
    v, i = torch.topk(logits, k)
    min_topk = v[-1]
    return torch.where(logits < min_topk, torch.tensor(-1e10, device=logits.device), logits)

def generate_autoregressive(model, start_tokens, max_new_tokens=50, temperature=1.0, top_k=0):
    # start_tokens: Tensor shape (start_len, batch)
    model.eval()
    generated = start_tokens.clone().to(next(model.parameters()).device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)               # (cur_len, batch, vocab)
            next_logits = logits[-1, :, :]          # (batch, vocab)
            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                # apply top-k per batch row
                next_tokens = []
                for b in range(next_logits.size(0)):
                    logits_b = top_k_logits(next_logits[b], k=top_k)
                    probs = F.softmax(logits_b, dim=-1)
                    next_tokens.append(torch.multinomial(probs, num_samples=1))
                next_token = torch.cat(next_tokens, dim=0).unsqueeze(0)  # (1, batch)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).t()  # (1, batch)
            generated = torch.cat((generated, next_token), dim=0)
            # keep context window
            if generated.size(0) > model.max_len:
                generated = generated[-model.max_len :, :]
    return generated  # (total_len, batch)

# --------------------------
# Training loop (toy)
# --------------------------
print("Starting training (toy). Adjust hyperparams for serious training.")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for it in range(iters_per_epoch):
        x, y = get_random_batch(batch_size, seq_len)  # shapes (seq_len, batch)
        optimizer.zero_grad()
        logits = model(x)  # (seq_len, batch, vocab)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / iters_per_epoch
    print(f"Epoch {epoch:3d} | avg loss {avg_loss:.4f} | time {time.time()-t0:.2f}s")
    # sample a short generation for quick sanity check
    start = tokens[: min(6, len(tokens)) ]
    start_idx = torch.tensor([[stoi[w] for w in start]], dtype=torch.long).t().to(device)  # (start_len, 1)
    sample = generate_autoregressive(model, start_idx, max_new_tokens=40, temperature=0.9, top_k=5)
    sample_txt = " ".join([itos[int(i)] for i in sample[:, 0].tolist()])
    print(" sample:", sample_txt)

# --------------------------
# Save / inference example
# --------------------------
torch.save({"model_state": model.state_dict(), "stoi": stoi, "itos": itos}, "mini_gpt.pth")
print("Saved checkpoint: mini_gpt.pth")

# Generate final output
start = tokens[: min(8, len(tokens)) ]
start_idx = torch.tensor([[stoi[w] for w in start]], dtype=torch.long).t().to(device)
out = generate_autoregressive(model, start_idx, max_new_tokens=max_generate, temperature=1.0, top_k=0)
print("Final generation:")
print(" ".join([itos[int(i)] for i in out[:, 0].tolist()]))