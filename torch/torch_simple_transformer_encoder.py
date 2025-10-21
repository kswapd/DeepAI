import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Multi-Head Self Attention ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # (batch, seq_len, embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [B, N, num_heads, head_dim]

        # attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        # weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Residual 1
        x = x + self.mlp(self.norm2(x))    # Residual 2
        return x


# --- Simple Transformer Encoder ---
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))  # max seq_len = 100
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)  # for language modeling

    def forward(self, x):
        B, N = x.shape
        x = self.embed(x) + self.pos_embed[:, :N, :]
        x = self.blocks(x)
        return self.head(x)


# --- Demo Run ---
if __name__ == "__main__":
    model = SimpleTransformer(vocab_size=1000, embed_dim=128, depth=2, num_heads=4)
    x = torch.randint(0, 1000, (2, 20))  # batch=2, seq_len=20
    out = model(x)
    print("Output shape:", out.shape)  # (2, 20, vocab_size)