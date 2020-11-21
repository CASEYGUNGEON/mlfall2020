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
print(len(arr))
# normalize
minVal = np.zeros(12)
maxDif = np.zeros(12)
    # get minimum and range for each factor & result
for i in range(0,12):
    minVal[i] = arr[:,i].min()
    maxDif[i] = arr[:,i].max() - minVal[i]
        # perform normalization on each factor in each datum
    for x in range(0,len(arr)):
        arr[x,i] = (arr[x,i] - minVal[i]) / maxDif[i]


# separate X and Y values for easier use
X = np.zeros((len(arr),11))
Y = np.zeros(len(arr))
X = arr[:,0:10]
Y = arr[:,11]

# pick random sample for training
numTrainSamples = 2000
trainPicks = np.random.randint(0,len(arr),numTrainSamples)

# initialize arrays for training and test sets
XTrain = np.zeros((numTrainSamples,10))
YTrain = np.zeros(numTrainSamples)
XTest = np.zeros((len(arr) - numTrainSamples,10))
YTest = np.zeros(len(arr) - numTrainSamples)

print(len(XTest))

# separate training and test samples
curTrainIndex = 0
curTestIndex = 0
for i in range(0,len(arr) - 1):
    if i in trainPicks:
        print("add to train: " + str(curTestIndex) + " + " + str(i))
        XTrain[curTrainIndex] = X[i]
        YTrain[curTrainIndex] = Y[i]
        curTrainIndex += 1
    else:
        print("add to test: " + str(curTestIndex) + " + " + str(i))
        XTest[curTestIndex] = X[i]
        YTest[curTestIndex] = Y[i]
        curTestIndex += 1

#print(XTrain)
#print(YTrain)
#print(XTest)
#print(YTest)