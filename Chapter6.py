import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

#Chapter 6 Using a Neural Network to fit data 

#Linear functions are fine w/ handling linear relationships (n inputs variables, 1 output) Real relationships are complex and operate on multiple planes,m just 1 
#Stacking multiple neurons in a sequence order 1->2->3->4 simplifies to a linear relationship anyways 
#To handle non linear functions, we need activition functions 

#3 Main Activition Functions 

#Sigmoid (function image here) Maps input to (0,1) -> can cause vanishing gradients with 0 
#Tanh - Maps inputs on a range between [-1,1] centers at zero and saturates at the extremes //Saturates values on both , +1000 becomes 0.99
#Relu max(0,x) -> modern default //Doesn't saturate +100 


#6.1 Artical Neurons

x = torch.tensor([-3.0,-1.0, 0.0, 1.0, 3.0])

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
relu = nn.ReLU()

print(sigmoid(x))
print(tanh(x))
print(relu(x))


#6.2 Torch's nn module 

#nn.Module is the base class for the module: Provides torch.nn
#Parameter management, forward computation, state tracking 
#Use __call__ instead of forward -> model(x) instead of model.forward(x) -> __call__ calls the pre and post hooks 

# 2. Instantiate a linear model and loss function
#1 input feature, 1 output feature 
linear_model = nn.Linear(1,-1)
loss_fn = nn.MSELoss()

#6.3 Finally a Neural Network 

#Non linear networks require #nn.Sequential to mkae chains

#In this context, the number 13 is arbitary 
#Input(1) -> Linear(1,13) -> Tanh() -> Linear(13,1) -> Output(1)

# 1. Define network using OrderedDict for readable layer names
seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 13)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(13, 1))
]))


print("Model architecture:\n", seq_model)