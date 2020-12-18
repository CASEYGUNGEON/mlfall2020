# Standard Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#########################################################################
# Loading in data
data = pd.read_csv('.\\winequality-white.csv', sep = ';')
arr = pd.DataFrame(data).to_numpy()

#########################################################################
# Normalize Data
minval = np.zeros(13) 
dif = np.zeros(13)    
for i in range(0,12): # for each category
    minval[i] = arr[1:,i].min() # get min
    dif[i] = arr[1:,i].max() - minval[i] # get difference
    for j in range(1,len(arr)): # for each value in each category
        arr[j,i] = (arr[j,i] - minval[i]) / dif[i] # normalizing data

# Assign values for X and Y
Xvals = np.ones((len(arr)-1,12))
Yvals = np.zeros(len(arr)-1)

n = len(arr)-1
S = np.random.permutation(n) 

for i in range(0,n): # randomizing data
    Xvals[i,1:12] = arr[S[i]+1,0:11] # Xvals[:, 0] is 1, fill in the rest with the X values from the dataset
    Yvals[i] = arr[S[i]+1,11] # fill in Y values from dataset

#########################################################################
# Number of training samples
TrainNum = 2000 # Can be changed

# Splitting data into testing and training 

# Training Samples 
Xtrain = np.zeros((TrainNum,12))
Ytrain = np.zeros(TrainNum)
Xtrain[:, :] = Xvals[0:TrainNum, :]
Ytrain[:] = Yvals[0:TrainNum]

# Testing Samples
Xtest = np.zeros((len(arr)-TrainNum-1,12))
Ytest = np.zeros(len(arr)-TrainNum-1)
Xtest[:, :] = Xvals[TrainNum:len(arr), :]
Ytest[:] = Yvals[TrainNum:len(arr)]

#########################################################################
# Gradient Descent Method
print("Starting Training")

numIterations = 200
alpha = 0.1

m = len(Ytrain)
arrCost = [];
XT = np.transpose(Xtrain)
theta = np.zeros(12)

# Finding Theta via Gradient Descent Method
for iterations in range(0, numIterations):
    sum = np.zeros(12)
    for i in range(0,m):
        dot = theta.dot(XT[:,i]) # intermediate calculation
        for j in range(0,12):
            sum[j] = (dot - Ytrain[i]) * XT[j,i] + sum[j] # GD
    sum[:] = alpha/(m) * sum[:]
    theta = np.subtract(theta, sum)

    # Getting Cost Function 
    sum1 = 0
    for i in range(0,m):
        sum1 = (theta.dot(XT[:,i]) - Ytrain[i])**2 + sum1
    
    atmp = sum1/2/(m)
    arrCost.append(atmp)

print("Done training")

##########################################################################
# Test how well the predictions and ground-truth values match
print("Start testing")

testNum = len(Ytest)
L1Err = L2Err = STDErr = mean = 0

predictions = np.zeros(testNum)
Error = np.zeros(testNum)
TransposedTesting = np.transpose(Xtest)

for i in range(0,testNum):
    predictions[i] = theta.dot(TransposedTesting[:,i])
    L1Err = abs(predictions[i] - float(Ytest[i])) + L1Err # uses absolute value
    L2Err = (predictions[i] - float(Ytest[i]))**2 + L2Err # uses square of error
    Error[i] = predictions[i] - float(Ytest[i]) # for calculating standard deviation
    mean = Error[i] + mean

mean = mean/testNum
for i in range(0,testNum):
    STDErr = abs(Error[i] - mean)**2 + STDErr

STDErr = math.sqrt(STDErr / testNum)
L1Error = L1Err/testNum
L2Error = L2Err/testNum

print('L1 Error: {}'.format(L1Error))
print('L2 Error: {}'.format(L2Error))
print('Standard Deviation: {}'.format(STDErr))

print("Done testing")

##########################################################################
# Plotting results of cost function from training

plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {} theta = {}'.format(alpha, theta))
plt.show()