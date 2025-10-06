import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Hyperparameters
vocab_size = 20      # small vocabulary
d_model = 16         # embedding size
seq_len = 5          # sequence length
nhead = 4            # attention heads
num_layers = 2       # number of decoder layers
batch_size = 2
epochs = 200
def generate_sequence(model, start_tokens, memory, max_len=seq_len):
    """
    start_tokens: [start_seq_len, batch_size]
    memory: encoder memory (can be zeros for GPT-style)
    """
    model.eval()
    generated = start_tokens.clone()
    
    for _ in range(max_len - start_tokens.size(0)):
        logits = model(generated, memory)             # [seq_len, batch, vocab]
        next_token_logits = logits[-1, :, :]         # last token
        next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1)
        next_token = next_token.unsqueeze(0)         # shape: [1, batch]
        generated = torch.cat((generated, next_token), dim=0)
    
    return generated

def generate_data(batch_size, seq_len, vocab_size):
    data = torch.randint(0, vocab_size, (seq_len, batch_size))
    target = (data + 1) % vocab_size
    return data, target


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        #print(self.decoder_layer)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt_seq, memory):
        # tgt_seq: [seq_len, batch_size]
        tgt_emb = self.embedding(tgt_seq)  # [seq_len, batch_size, d_model]

        # Generate mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(0))
        
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out = self.fc_out(out)  # logits over vocabulary
        return out






model = TinyGPT(vocab_size, d_model, nhead, num_layers)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
# Random "source" memory (from encoder, or can be zeros for GPT-style)
memory = torch.zeros(seq_len, batch_size, d_model)


# Training data example
src, tgt = generate_data(batch_size, seq_len, vocab_size)
print("Source:\n", src)
print("Target:\n", tgt)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    src, tgt = generate_data(batch_size, seq_len, vocab_size)
    
    logits = model(src, memory)  # [seq_len, batch, vocab]
    w_before = model.transformer_decoder.layers[0].self_attn.in_proj_weight.clone()
    # reshape for cross-entropy: [seq_len*batch, vocab]
    loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
    loss.backward()
    optimizer.step()
    w_after = model.transformer_decoder.layers[0].self_attn.in_proj_weight
    print(torch.allclose(w_before, w_after)) 
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


start_tokens = torch.randint(0, vocab_size, (1, batch_size))
print("Started tokens:")
print(start_tokens)
# Use the same autoregressive generation function from before
generated_seq = generate_sequence(model, start_tokens, memory, max_len=seq_len)
print("Generated sequence after training:")
print(generated_seq)