import torch
import torch.nn as nn

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    print("Accelerator is available:", torch.accelerator.current_accelerator())
    tensor = tensor.to(torch.accelerator.current_accelerator())

print(f"Device tensor is stored on: {tensor.device}")