import torch
import torch.nn as nn
import torch.optim as optim
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)  # [src_len, batch, emb_dim]
        outputs, hidden = self.rnn(embedded)
        return hidden  # only return final hidden state
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        hidden = self.encoder(src)
        input = trg[0, :]  # first token <sos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
    

    
# toy parameters
INPUT_DIM = OUTPUT_DIM = 10
EMB_DIM = 16
HID_DIM = 32
LR = 0.01

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
model = Seq2Seq(enc, dec)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# fake data (reverse sequence)
for epoch in range(200):
    src = torch.randint(1, INPUT_DIM, (5, 2))   # [seq_len=5, batch=2]
    trg = torch.flip(src, [0])                  # reversed target
    optimizer.zero_grad()
    output = model(src, trg)
    output_dim = output.shape[-1]
    loss = criterion(output[1:].view(-1, output_dim), trg[1:].reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")
