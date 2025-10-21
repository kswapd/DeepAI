import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Multi-Head Causal Self-Attention ---
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        # q, k, v = qkv.unbind(dim=2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # ---- causal mask: block future tokens ----
        # mask = torch.tril(torch.ones(N, N, device=x.device))
        mask = torch.tril(torch.ones(N, N, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# --- Transformer Decoder Block ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- Decoder-only Transformer Language Model ---
class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, depth=2, num_heads=4, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, N = idx.shape
        x = self.embed(idx) + self.pos_embed[:, :N, :]
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)  # [B, N, vocab_size]
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
    

# ---- Training Demo ----
vocab_size = 1000
model = SimpleTransformerDecoder(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(200):
    # random dummy batch
    x = torch.randint(0, vocab_size, (4, 20))     # input tokens
    # y = x[:, 1:]                                 # next-token targets
    # x = x[:, :-1]
    y = (x + 1) % vocab_size
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step:3d} | loss {loss.item():.4f}")


# ---- Inference ----
context = torch.randint(0, vocab_size, (1, 5))  # 5 starting tokens
generated = model.generate(context, max_new_tokens=10)
print("Input:", context.tolist())
print("Generated:", generated.tolist())