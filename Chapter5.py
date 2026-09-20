import torch
import torch.optim as optim 

#Chapter 5 The Mechanics of Learning 

#5.1 Timeless lesson in modeling 

#Machine Learning process goes through a 4 step process 

#1. Find input / output variables for the real world 
#2. Formula some formula for turning inputs to outputs 
#3. Discover some way to measure error
#4 re-adjust the inputs to minimize error and repeat 

#5.2 Learning is just parameter evaulation 

#Look at converting celsuis to faherniet and vice versa
#We start with a linear formula    F = C * W(Weight) + B(Bias)
#Use Mean Squared Error to handle Error (big massive diffs between preidcted and real are penalized heavily ), and measure error and try to find ideal weight and bias 

#5.3 Less loss is what we want

#We need a scalar metric to measure the difference between between prediction and ground truth 

def loss_fn(t_p,t_c):
    square_diffs = (t_p - t_c) **2
    return square_diffs.mean()

#We initial dummy values for weight and bias and go from there

#5.4 Down along the gradient 


#t_u = temp using an unknown scale unknown (input) (fahenreit)
#t_p = temp predicted using the model
#t_c = temp celsius ground truth (output)

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
t_nu = t_u * 0.1

#The decrease the lose, parameters must be nudged, we can approximate the rate of change numerically by adding a tiny perturbation (delta) to a parameter
#Delta is a tiny artifical nudge used to approximate the derivative 
#Imagine we are blindfolded and at the bottom of a valley. we want to know if the ground sloping up or down in front of us. We take a step forward of 1 ft (delta) 
#we didn't walk anywhere, we just approximated the slop (derivate) at distance self + delta 

#rateFormula.png

#The rateFormula will tell us if we should increase or decrease the delta parameter 

#This rate formula is fine when dealing with a small amount of variables. Scales poorly when dealing with a large amount of variables. 
#We use the chain rule to calc derivatives 

#Derivative of the Mean Squared Error (MSE) w/ respect to t_p 
#How much the loss would increase or decrease if the predicted value changed by a tiny amount 
def dloss_fn(t_p,t_c):
    return 2 * (t_p -t_c) / t_p.size(0)


#The derivate of the model with respect to weight w
#For t_p = t_u * w + b\
#If I move w by 1 unit, how much does t_p change 
def dmodel_dw(t_u,w,b):
    return t_u


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


#Learning Rate

#Gradient shares which direction is downhill / uphill and by how much, doesn't share how big of a step to take 
#Learning rate tells us how big of a step to take to reach the bottom of the hill
#Low learning rate means it takes longer to reach the bottom, but we probably won't overshoot it


#Manual Training Loop 

def model(t_u, w, b):
    return w * t_u + b

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(n_epochs):
        w,b = params
        t_p = model(t_u,w,b)

        loss = loss_fn(t_p,t_c)
        gradient = grad_fn(t_u,t_c,t_p,w,b)

        params = params - learning_rate * gradient

    return params




#5.5 PyTorch's autograd: Backprogage all things
#Manually calcing the gradient on every tensor can get expensive and time consuming 
# Pytorch fixes this slightly by adding an "autograd" parameter for tensors, using a DAG, it will save all operates applied to a tensor 
#all tensors have a grad attribute anyways, its false by default 

#Forward Pass -> Taking data from the left to right
#inputs -> layers -> prediction -> loss
#Takes the inputs and plassing through layers, figuring out which neurons activite and the overall prediction and calculates the loss
#Under the hood (if autograd is enabled) = Each layer will keep track of all operations that are done on the neuron 


#Backwards Pass -> Taking the data from the right to the left 
#Who is the blame for the loss
#Output Layer = How much its weights contributed to the loss
#Hidden Layer = How much each neuron's activition / lack of activition contributed to the result AND How much its weights contributed to the activition 
#Trace its way back to the input 

#Each parameter is left with a specific gradient value stroed in .grad that answers: "If I nudge this specific parameter up or down, will the final loss increase or decrease, and by how much?"
#After the backward pass is done, the optimizer can step in and adjust the parameters weights ion the optimizer.step()




params = torch.tensor([1.0,0.0],requires_grad=True)

print("Step 1. Grads before Forward Pass")
print(params.grad)

#loss = loss_fn(model())

#Forward Pass
t_p = model(t_nu,*params)

loss = loss_fn(t_p,t_c)

print(f"\n2. Computed Loss: {loss.item():.4f}")
# 4. Backward pass: traverses the graph and populates .grad
loss.backward()



print("\n3. params.grad AFTER loss.backward():")
print(params.grad)


#zeroing gradients is needed after each epoch to ensure each training session is done in isolation without any data drift  

#5.6 Optimizers 

#Vanilla Gradient Descent = subtracting learning_rate*grad
#Optimizers are a whole field in of itself 

#To setup an optimizer 
params = torch.tensor([1.0,0.0], requires_grad=True)
learning_rate = 1e-5

#SGD = Stochastic Gradient Descent 
#This acts like a vanilla gradient descent when the momentum is set to zero which is the default 
#Stochastic in this context means that it is averaged from a random set of inputs (mini batch)
#Stochastic also means it can not be predicted with total certainty 
optimizer = optim.SGD([params], lr = learning_rate)


t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()

optimizer.step()

print(params)



#5.8 Summary 

#Linear models are the easiest to work w/ -> less dimensions 
#SGD can be used for parameter estimation 
#Rate of change of the loss function can be used to change the parameters 
#Optimezers can be used to automatelly update the parameters to minimize loss based on the grad DAG from the tensor 
#Overfitting can happen when the model is tooo good on training data





