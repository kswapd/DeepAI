import torch
import torch.nn as nn
import torch.optim as optim

# 假设词表大小 = 10，嵌入维度 = 4
vocab_size = 10
embed_dim = 4

# 1. 定义模型：只有一个 embedding + 线性层
class TinyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)   # 词嵌入矩阵
        self.fc = nn.Linear(embed_dim, vocab_size)         # 输出层，预测下一个 token

    def forward(self, x):
        x = self.embed(x)   # 查 embedding
        x = self.fc(x)      # 预测下一个词
        return x

# 2. 初始化模型
mmmm = TinyLM(vocab_size, embed_dim)

# 3. 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mmmm.parameters(), lr=0.05)

# 4. 伪造训练数据（每个样本：输入 token → 预测下一个 token）
# 比如 token 2 后面应该接 3
inputs = torch.tensor( [2, 5, 7, 8, 9])      # 输入
targets = torch.tensor([3, 6, 8, 9, 9])     # 目标 (下一个 token)
print("\nEmbedding matrix before training:")
print(mmmm.embed.weight.data)
# 5. 训练几轮
for epoch in range(10):
    optimizer.zero_grad()
    outputs = mmmm(inputs)
    loss = criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6. 看 embedding 矩阵有没有更新
print("\nEmbedding matrix after training:")
print(mmmm.embed.weight.data)