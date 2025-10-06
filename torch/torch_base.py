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

t3 = torch.zeros((2,3), dtype=torch.long)
t4 = torch.cat((t3, torch.tensor([[1,2,3]])), dim=0)
t5 =torch.cat((t3, torch.tensor([[1,2],[3,4]])), dim=1)
t6 =t3.unsqueeze(0) 
t7 =torch.cat((t6, torch.tensor([[[1,2],[3,4]]])), dim=2)
print(f"t3: {t3.shape}, {t3.dtype}, t3: {t3}")
print(f"t4: {t4.shape}, {t4.dtype}, t4: {t4}")
print(f"t5: {t5.shape}, {t5.dtype}, t5: {t5}")
print(f"t6: {t6.shape}, {t6.dtype}, t6: {t6}")
print(f"t7: {t7.shape}, {t7.dtype}, t7: {t7}")
t8 = torch.zeros((0,3), dtype=torch.long)
print(f"t8: {t8.shape}, {t8.dim()}")
t9 = torch.cat((t8, torch.tensor([[1,2,3]])), dim=0)
print(f"t9: {t9.shape}, {t9.dim()}")
t10 = torch.zeros(0, dtype=torch.long)
print(f"t10: {t10.shape}, {t10.dim()}")
t11 = torch.cat((t10, torch.tensor([1,2,3])), dim=0)
print(f"t11: {t11.shape}, {t11.dim()}")