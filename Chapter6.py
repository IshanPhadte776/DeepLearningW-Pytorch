import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]).unsqueeze(1)
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)
t_nu = t_u * 0.1


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
# linear_model = nn.Linear(1,-1)
# loss_fn = nn.MSELoss()

#6.3 Finally a Neural Network 

#Non linear networks require #nn.Sequential to mkae chains

#In this context, the number 13 is arbitary 
#Input(1) -> Linear(1,13) -> Tanh() -> Linear(13,1) -> Output(1)

#A singular neuron can learn a linear function. n values input, 1 value output (Regression)

#More neurons = ability to grasp more complex relationships between input and output 
#Every neuron we add provides the network with another independent mathematical joint or threshold, allowing it to bend and fold its predictions to capture more intricate curves and transitions. In higher dimensions, having more neurons allows the model to slice, dice, and partition the complex multi-variable space from countless different angles simultaneously. Ultimately, this expanded pool of parameters serves as the "building blocks" needed to detect, combine, and reconstruct elaborate patterns that a smaller network would be forced to oversimplify.


# 1. Define network using OrderedDict for readable layer names
seq_model = nn.Sequential(OrderedDict([
    #1 Input Node, 11 Output Nodes
    ('hidden_linear', nn.Linear(1, 11)),
    ('hidden_activation', nn.Tanh()),
    #11 Inputs Nodes, 1 Output Node
    ('output_linear', nn.Linear(11,5))
]))


print("Model architecture:\n", seq_model)

print("\nInspecting parameters: ")

for name, param in seq_model.named_parameters():
    print(f" {name:20} -> Shape : {list(param.shape)}")

optimizer = optim.SGD(seq_model.parameters(), lr= 1e-3)

loss_fn = nn.MSELoss()

for epoch in range (1,5001):
    t_p = seq_model(t_nu)
    loss = loss_fn(t_p, t_c)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print("Epoch " + str(epoch) + "Loss: " + str(loss))

    

#6.5 Exercises 

#What happens whe we add more linear output from the model? 
#Multiple neurons added in sequence squash the model into a linear equations, really doesn't change much in-regards to the complexity of relationship we can express with the model

#Can we get the model to overfit our data. 
#Yes, caused by providing too little data, and running too much training on the input. L1 / L2 Regularization can help with this issue 