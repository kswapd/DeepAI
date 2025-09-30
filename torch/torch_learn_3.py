import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_kernels(weights, title):
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            ax.imshow(weights[i, 0].cpu().detach().numpy(), cmap="gray")
            ax.axis("off")
    plt.show()
class MyNet(nn.Module):
    def __init__(self,  in_channels=1, img_width=20, img_height=20, num_classes=10):
        super(MyNet, self).__init__()
        self.pool_size = 2
        self.kernel_size = 3
        self.channel_size = 16
        self.out_frame_size = int(self.channel_size*img_width/self.pool_size*img_height/self.pool_size)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.channel_size, kernel_size=self.kernel_size, padding=1)
        self.pool = nn.MaxPool2d(self.pool_size, self.pool_size)  # 尺寸减半 28->14
        #self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 池化后尺寸 14->7
        self.fc1 = nn.Linear(self.out_frame_size, num_classes)  # 32个通道，每个7x7
    def forward(self, x):
        x = x / 100
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.out_frame_size)
        x = self.fc1(x)
        preds = torch.max(x, dim=1)
        return x, preds
    
class MLP(nn.Module):
    def __init__(self,  in_channels=1, img_width=20, img_height=20, num_classes=10):
        super().__init__()
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.fc1 = nn.Linear(img_width*img_height, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, num_classes)  # 二分类

    def forward(self, x):
        x = x.view(x.size(0), -1)/100  # 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        preds = torch.max(x, dim=1)
        return x, preds
in_channels = 1
img_width = 20
img_height = 20
num_classes = 10
batch_size = 10
model = MyNet(in_channels=in_channels, img_width=img_width, img_height=img_height, num_classes=num_classes)
#model = MLP(in_channels=in_channels, img_width=img_width, img_height=img_height, num_classes=num_classes)
#x = torch.tensor([[1.0, 2.0]])
x = torch.full((batch_size, in_channels, img_width, img_height), 1.0, dtype=torch.float32)
for i in range(x.shape[0]):
    x[i,0,:,:] = float(i % num_classes) * 10
    #print("x[%d]", x[i,0,:,:])
y = torch.tensor([0,1,2,3,4,5,6,7,8,9])


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
#plot_kernels(model.conv1.weight, "Conv1 Kernels (Before Training)")
#print("before:", model.conv1.weight)   
#print("\nEmbedding matrix before training:")
#print("权重:", model.fc.weight.data)

for epoch in range(200):
    #print(f"Epoch {epoch+1} started")
    optimizer.zero_grad()
    outputs, p = model(x)
    loss = criterion(outputs, y)
    #print(f"{epoch+1}: outputs:{outputs}, y: {y}, loss: {loss}, weight: {model.fc1.weight.data}")
    #print(f"{epoch+1}: loss: {loss}, p: {p}")
    loss.backward()
    optimizer.step()
    #print(f"Epoch {epoch+1} done")
    #print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    #print(f"权重: {epoch}", model.fc.weight.data)

# 6. 看 embedding 矩阵有没有更新
#print("\nEmbedding matrix after training:")
#print(mmmm.embed.weight.data)
logits, preds = model(x)
#print("更新后 logits:\n", logits)
#print("更新后 preds:\n", preds)
#plot_kernels(model.conv1.weight, "Conv1 Kernels (After Training)")
#print("after:", model.conv1.weight) 
#output = model(x)
#print(output)
#loss = (output - y).pow(2).mean()



