import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(SimpleRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        #self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        for name, param in self.rnn.named_parameters():
            print(name, param.shape)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        out = out[:, -1, :]      # last time step
        return self.fc(out)

# Create model, loss, optimizer
model = SimpleRNNClassifier(vocab_size=5000, embed_dim=100, hidden_dim=128, num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Example dataset
# Each row = sentence of word IDs
X = torch.randint(0, 5000, (1000, 20))  # 1000 samples, each 20 tokens long
y = torch.randint(0, 2, (1000,))        # binary labels (0 or 1)

# Split into training & validation
train_ds = TensorDataset(X[:800], y[:800])
val_ds = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)



num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            preds = model(val_x)
            predicted = preds.argmax(1)
            correct += (predicted == val_y).sum().item()
            total += val_y.size(0)
    
    acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")