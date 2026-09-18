import torch

#Creates a Tensor w/ 1's by default
#Both ptorch and numpy are R,C libraries 
#inlike python lists where the elements are in scattered memory, numpy / torch tensors are in contineous memory
#the values in the tensor are floats
a = torch.ones(3)

print(a)
print(a[1])

b = torch.zeros(3)

print(b)

#We can pass a python list as well
c = torch.tensor([1,2,3,4])
print(c)

#Provides the shape of the tensor
print(c.shape)

##3.3 Indexing Tensors