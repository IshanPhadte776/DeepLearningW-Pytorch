import torch
import imageio
import imageio.v3 as iio
import csv

#Chapter 4 Real world data representation using tensors 

#4.1 Working with images 
#Images are stored in grids w/ color intendisities 

#OpenCV / Numpy use HWC order [Height, Width, Channels]
#Pytorch uses CHW [ Channels, Height,Width] or BCHW [Batch_size, Channels, Height, Width]
#rn, img_arr is a numpy like array
img_arr = imageio.imread("totkbackground.jpg")

print(img_arr.shape)

img = torch.from_numpy(img_arr)
#moves the dimensions of the array
out = img.permute(2,0,1)

#Normaliztion of data 

#This is a placeholder for a batch of 3 images
batch_size = 3
#B,C,H,W for Pytorch
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

#print(batch)

batch = batch.float()
batch /= 255.0 

print(batch)

#4.2 3D Images: Volumetric data

#1 dcm file = 1 2D Slice [512,512]
#We need all of those dcm files for a medical scan 
dir_path = "data/p1ch4/volumetric-dicom/"
vol_arr = imageio.volread(dir_path, 'DICOM')


#unsqueeze() inserts a new dimension of size 1 at a specific position within the tensor 

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol,0)

print(vol.shape)


#4.3 Representing Tabular data

#Heterogeneous Structures vs Homogeneous Tensor 
#CSV Files = Different types, pytorch tensors all have the same datatype 
#However, numpy can work with csv, torch can't 


#4.4 Working with Time Series Data 
#Time series problems track tabular or sensory variables sequentially across continuous time, we add a time dimension 
#Layout looks like [Batch size, Features, Time_steps]

#4.5 Representing Text 
#Fundamentally -> Natural Language lacks direct numerial coordinates. Words and sentences must be mapped to symbols and structures and into numerial spaces while presenting the semantic meaning

#Character Level REpresentation

#We can have a mapping between integer and letter in alphabet 
#Vocab construction = have an unordered map / set with all indexes of words {(124),(14532)} 
#Sentences become matrix of size  Length * sizeOfVocab 
#Lacks Semantic meaning

#Word Level Representation
#Instead of letters, use words as the base unit of measurement
# do preprocessing 
# Map words to integer index for vocab
# 
 
#Sparse Matrix Problem 
#Any of these previosu approach are ineffienct, word embeddings is the solution to get semantic meaning and handle adjacanety of words

#Word Embeddings
#The typically apporach, each word is mapped to an embedded 


#4.7 Exercises 
# 
#  
dir_path_blue = "data/p1ch4/coloured-photos/blueimage.jpg"
dir_path_green = "data/p1ch4/coloured-photos/greenapple.jpg"
dir_path_red = "data/p1ch4/coloured-photos/redapple.jpg"

blue_arr = iio.imread(dir_path_blue)
green_arr = iio.imread(dir_path_green)
red_arr = iio.imread(dir_path_red)

blue_tensor = torch.from_numpy(blue_arr)
green_tensor = torch.from_numpy(green_arr)
red_tensor = torch.from_numpy(red_arr)


#I learnt we can't get the mean from int based tensors
# We can only get mean from floating point
#We must cast to float
#dim = only include speciifc dimensions 
print(blue_tensor.float().mean(dim=(0), keepdim=False))
print(green_tensor.float().mean(dim=(0), keepdim=False))
print(red_tensor.float().mean(dim=(0), keepdim=False))
