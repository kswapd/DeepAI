import torch
import torch.nn as nn
import torch.optim as optim

class MyNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()                        # 激活函数
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        #print("权重:", self.fc.weight.data)
        #print("偏置:", self.fc.bias.data)
    def forward(self, x):
        #return self.fc(x)
        #logits = self.fc(x)                        # 直接输出 logits
        #preds = torch.argmax(logits, dim=1)        # 预测类别
        #return logits, preds
        hidden = self.relu(self.fc1(x))           # 隐藏层输出
        logits = self.fc2(hidden)                 # 输出 logits
        preds = torch.argmax(logits, dim=1)       # 预测类别
        return logits, preds

model = MyNet(2,2)
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([1])


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

#print("\nEmbedding matrix before training:")
#print("权重:", model.fc.weight.data)

for epoch in range(10):
    optimizer.zero_grad()
    outputs, p = model(x)
    loss = criterion(outputs, y)
    print(f"outputs:{outputs}, y: {y}, loss: {loss}, weight: {model.fc1.weight.data}")
    loss.backward()
    optimizer.step()

    #print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    #print(f"权重: {epoch}", model.fc.weight.data)

# 6. 看 embedding 矩阵有没有更新
#print("\nEmbedding matrix after training:")
#print(mmmm.embed.weight.data)
logits, preds = model(x)
print("更新后 logits:\n", logits)
print("更新后 preds:\n", preds)

#output = model(x)
#print(output)
#loss = (output - y).pow(2).mean()



