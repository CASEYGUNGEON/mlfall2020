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
minval = np.zeros(12)
maxdif = np.zeros(12)
for i in range(0,12): #get minimum and range for each factor & result
    minval[i] = arr[:,i].min()
    maxdif[i] = arr[:,i].max() - minval[i]
    for x in range(0,len(arr)): # perform normalization on each factor in each datum
        arr[x,i] = (arr[x,i] - minval[i]) / maxdif[i]

X = np.zeros((len(arr),11))
Y = np.zeros(len(arr))
X = arr[:,0:10]
Y = arr[:,11]