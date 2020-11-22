# Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# from download_data import download_data # Not using this
# from GD import gradientDescent
# from dataNormalization import rescaleMatrix

#########################################################################
# Loading in data
# The file location matters, may either need to change where it's currently located
# NOTE: Must delete first row of the .csv files to skip parsing data
data = pd.read_csv('~\\Documents\\GitHub\\mlfall2020\\winequality-white.csv', sep=';')
arr = pd.DataFrame(data).to_numpy()
# print(arr) # for testing

#########################################################################
# Normalize Data

minval = np.zeros(13) 
dif = np.zeros(13)    
for i in range(0,12): # for each category
    minval[i] = arr[1:,i].min() # get min
    dif[i] = arr[1:,i].max() - minval[i] # get difference
    for j in range(1,4898): # for each value in each category
        arr[j,i] = (arr[j,i] - minval[i]) / dif[i] # normalizing data

# print(arr) # for testing

# Assign values for X and Y
Xvals = np.ones((4898,12))
Yvals = np.zeros(4898)
Xvals[:, 1:12] = arr[1:, 0:11] # Xvals[:, 0] is 1, fill in the rest with the X values from the dataset
Yvals[:] = arr[1:, 11] # fill in Y values from dataset

#print(Xvals)
#print(Yvals)

#########################################################################
# Number of training samples
# Can be changed
TrainNum = 2000 

# Splitting data into testing and training 
Xtrain = np.zeros((TrainNum,12))
Ytrain = np.zeros(TrainNum)
Xtrain[:, :] = Xvals[0:TrainNum, :]
Ytrain[:] = Yvals[0:TrainNum]

# print(Xtrain) 
# print(Ytrain)

Xtest = np.zeros((4897-TrainNum+1,12))
Ytest = np.zeros(4897-TrainNum+1)
Xtest[:, :] = Xvals[TrainNum:4897+1, :]
Ytest[:] = Yvals[TrainNum:4897+1]

# print(Xtest)
# print(Ytest)

#########################################################################
# Gradient Descent Method
print("Starting Training")

numIterations = 1000
alpha = 0.1

m = len(Ytrain)
arrCost = [];
XT = np.transpose(Xtrain)
theta = np.zeros(12)
#[a,b] = XT.shape
#print(a)
#print(b)
# Finding Theta
for iterations in range(0, numIterations):
    sum = np.zeros(12)
    #print(iterations)
    for i in range(0,m):
        dot = theta.dot(XT[:,i])
        for j in range(0,12):
            sum[j] = (dot - Ytrain[i]) * XT[j,i] + sum[j]
            #print(XT[j,i])
    sum[:] = alpha/(m) * sum[:]
    theta = np.subtract(theta, sum)

# Getting Cost Function 

    sum1 = 0
    for i in range(0,m):
        sum1 = (theta.dot(XT[:,i]) - Ytrain[i])**2 + sum1
    atmp = sum1/2/(m)
    #print(atmp)
    arrCost.append(atmp)
print("Done training")

##########################################################################
# Test how well the predictions and ground-truth values match
print("Start testing")
testNum = len(Ytest)
print(testNum)

predictions = np.zeros(testNum)
Error = np.zeros(testNum)
TransposedTesting = np.transpose(Xtest)
#print(np.shape(TransposedTesting))
L1Err = L2Err = STDErr = mean = 0
for i in range(0,testNum):
    predictions[i] = theta.dot(TransposedTesting[:,i])
    L1Err = abs(predictions[i] - float(Ytest[i])) + L1Err # uses absolute value
    L2Err = (predictions[i] - float(Ytest[i]))**2 + L2Err # uses square of error
    Error[i] = predictions[i] - float(Ytest[i]) # for calculating standard deviation
    mean = Error[i] + mean
   #print(i)

mean = mean/testNum
for i in range(0,testNum):
    STDErr = abs(Error[i] - mean)**2 + STDErr

STDErr = math.sqrt(STDErr / testNum)
L1Error = L1Err/testNum
L2Error = L2Err/testNum
print(L1Error)
print(L2Error)
print(STDErr)
print("Done testing")

##########################################################################
# Plotting results of cost function from training
# NOTE: all plots must be added to the end of the program, otherwise any code
#       afterwards may not run

plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {} theta = {}'.format(alpha, theta))
plt.show()