import torch
import torch.nn as nn

myTensor = torch.rand(3,4,2)
myTensor2 = torch.rand(3,2,5)
print (myTensor)
print(f"Shape of tensor: {myTensor.shape}, Datatype of tensor: {myTensor.dtype}, Device tensor is stored on: {myTensor.device}")
# We move our tensor to the current accelerator if available
#if torch.accelerator.is_available():
#    print("Accelerator is available:", torch.accelerator.current_accelerator())
#    myTensor = myTensor.to(torch.accelerator.current_accelerator())

myTensorT = myTensor.T
print(f"Shape of tensor: {myTensorT.shape}, Datatype of tensor: {myTensorT.dtype}, Device tensor is stored on: {myTensorT.device}")
myTensor = myTensor @ myTensor2
print (myTensor)

t1 = torch.rand(2,3,4, device='cuda')
t2 = torch.tensor([2,3,4])
print(f"t1: {t1.shape}, t2: {t2.shape}")