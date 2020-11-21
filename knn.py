#KNN model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# calculate distance between 2 points
def distance(x1, x2):
    dist = 0.0
    for i in range(len(x1)-1):
        dist += (x1[i] - x2[i])**2
    dist = sqrt(dist)
    return dist

# read data from csv

data = pd.read_csv('.\\winequality-white.csv', sep = ';')
arr = pd.DataFrame(data).to_numpy()

# normalize
for i in range(0,13): #get minimum and range for each factor & result
    minval[i] = arr[:,i].min()
    maxdif[i] = arr[:,i].max - minval[i]
    for x in arr.len(): # perform normalization on each factor in each datum
        arr[x,i] = (arr[x,i] - minval[i]) / maxdif[i]

