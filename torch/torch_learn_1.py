import torch
import torch.nn as nn
import torch.optim as optim
'''
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * x
z = y.sum()
z.backward()
print(x, y, z)
print(x.grad, y.grad, z.grad)
z.backward()
print(x, y, z)
print(x.grad, y.grad, z.grad)
print("ok")
'''

class MyNet(nn.Module):
    def __init__(self, input_dim=2, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        print("权重:", self.fc.weight.data)
        print("偏置:", self.fc.bias.data)
    def forward(self, x):
        #return self.fc(x)
        logits = self.fc(x)                        # 直接输出 logits
        preds = torch.argmax(logits, dim=1)        # 预测类别
        return logits, preds

model = MyNet(2,2)
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([1])


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

print("\nEmbedding matrix before training:")
print("权重:", model.fc.weight.data)

for epoch in range(10):
    optimizer.zero_grad()
    outputs, p = model(x)
    loss = criterion(outputs, y)
    print(f"outputs:{outputs}, y: {y}, loss: {loss}, weight: {model.fc.weight.data}")
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



