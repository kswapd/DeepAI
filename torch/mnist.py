import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 尺寸减半 28->14
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 池化后尺寸 14->7
        self.fc1 = nn.Linear(32*7*7, 10)  # 32个通道，每个7x7
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = self.fc1(x)
        return x
def plot_kernels(weights, title):
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            ax.imshow(weights[i, 0].cpu().detach().numpy(), cmap="gray")
            ax.axis("off")
    plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 网络、损失函数、优化器
net = MyNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
print("初始卷积核：")
plot_kernels(net.conv1.weight, "Conv1 Kernels (Before Training)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model is running in:", device)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
net.to("cpu")
start_time = time.time()
# 训练循环
for epoch in range(10):
    print(f"Epoch {epoch+1} started")
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")
end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")
print("训练后卷积核：")
plot_kernels(net.conv1.weight, "Conv1 Kernels (After Training)")