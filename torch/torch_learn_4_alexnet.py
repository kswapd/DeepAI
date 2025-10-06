import torch
import torchvision.models as models

# Load pretrained AlexNet
alexnet = models.alexnet(pretrained=True)
print(alexnet)