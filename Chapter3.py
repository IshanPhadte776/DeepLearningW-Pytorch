import torch
import numpy as np
#3.1 Tensors

#Creates a Tensor w/ 1's by default
#Both ptorch and numpy are R,C libraries 
#inlike python lists where the elements are in scattered memory, numpy / torch tensors are in contineous memory
#the values in the tensor are floats
a = torch.ones(3)

print(a)
print(a[1])

# 3.2 Tensors

b = torch.zeros(3)

print(b)

#We can pass a python list as well
c = torch.tensor([1,2,3,4])
print(c)

#Provides the shape of the tensor
print(c.shape)

##3.3 Indexing Tensors

#Start(inc), end (exc)
print(c[1:2])
#Start to end
print(c[0:])
#From start to 1 prior to the end
print(c[:-1])

#3.4 Named Tensors

#It can get confusing on which dimensions match up w/ what - batch size, channels, etc

#RGB, R,C
img = torch.randn(3,5,5)
print(img)

#weights_names = torch.tensor([0.1,0.4,0.6], names = ["channels"])

#3.5 Data Types

#Numbers in python are objects 
#Lists in Python are for collections of objects (Non contigoeus memory)
#Python's intercepter is slow compared to a compiled language

#torch.int
#torch.float64
#torch.float32
#Torch likes 64 bit floating points, more precision, but slowers and more resources
#16 it fp isn't offered on cpu but on gpu 

#to get data type
print(c.dtype)

#3.6 Tensor API

#Most operations can be done on both tensors of pytorch and on pytorch directly


d = torch.ones(3, 2)
d_t = torch.transpose(d, 0, 1)
print(a.shape)
print(d_t.shape)

#Groups of Tensor Commands

#Creation Ops - Creation of Tensors
#Slicing / Mutation Ops -> Slicing, Transposing 
#Random Sampling -> Randn 
#Parallelization -> For controlling the number of threads for CPU Ops
#Serialization -> Saving and Loading Tensors

#Math
#For Each
#Reducation 
#Comparison
#Transforming in the frquency domain 
#Operations for working with vectors 
#For Linear Algebra 


#3.7 / 3.8 Tensors: Inside Perspective 

#Tensors are made up from 2 sections. The contigoeus block of memory which stores the values, and the Tensor wrapper around it that has multidimensional index log on top of it
#Strides = The number of physical elements which must be skipped to reach the next valid address in memory
#Offset = how much off the pointer to access a specific variable from the base memory address

a.zero_() # the _ means in-place operation 

a.is_contiguous() #checks if the tensor is contigeous as some operations only work on contigoeus tensors 

#3.9 Moving Tensors to the GPU 

#Creates a torch tensor specifically for the device cuda
#points_gpu = torch.tensor([1,2,3], device='cuda')
#Moving a tensor to the gpu 
#a_gpu = a.to(device="cuda")
#Mvoing an operation to the gpu

#points_gpu = 2 * points_gpu.to(device= 'cuda')


#3.10 Numpy interoperability 

#Zero Copy Sharing
#The data isn't duplicated in memory, uses the same internal storage buffer for the data. 
#just rhe wrapper changes

points310 = torch.ones(3,4)

points_np = points310.numpy()
print(points_np)

#3.11 Generalized Tensors 

#There's Sparse tensors -> tensors with bunchs of zeros which are memory optimized
#Quantizied Tensors -> torch.qint8 for lightwieght inference deployments 


#3.12 Serializing Tensors 

#Pickle is python serialization library 
#Serialization can be used to save and load tensors to and frm files (checkpoints)

torch.save(points310, 'checkpoint.t')

#3.14 Exercises 

#1

listRange9 = torch.tensor(list(range(9)))

#Stride = 16
print(listRange9.stride())

#Offset = 0
print(listRange9.storage_offset())

#size = 9
print(listRange9.size())

#1a)

exercise1a = listRange9.view(3,3)

#3x3 matrix 
#view reshapes the matrix without copying the underlying data
print(exercise1a)


#1b)
#REmove the 0th row and column
exercise1b = exercise1a[1:,1:]
print(exercise1b)

#2
#Method can go on the object itself or via the torch module
print(listRange9.sqrt())

print(torch.sqrt(listRange9))

print(a.sqrt_())