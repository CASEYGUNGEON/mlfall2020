
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
data = pd.read_csv('.\\winequality-white.csv', sep = ';')
arr = pd.DataFrame(data).to_numpy()
# print(arr) # for testing

#########################################################################
# Normalize Data

# print('Normalizing Data')
minval = np.zeros(13) 
dif = np.zeros(13)    
for i in range(0,12): # for each category
    minval[i] = arr[1:,i].min() # get min
    dif[i] = arr[1:,i].max() - minval[i] # get difference
    for j in range(1,len(arr)): # for each value in each category
        arr[j,i] = (arr[j,i] - minval[i]) / dif[i] # normalizing data

# print(arr) # for testing

# Assign values for X and Y
Xvals = np.ones((len(arr)-1,12))
Yvals = np.zeros(len(arr)-1)

n = len(arr)-1
S = np.random.permutation(n) 

for i in range(0,n): # randomizing data
    Xvals[i,1:12] = arr[S[i]+1,0:11] # Xvals[:, 0] is 1, fill in the rest with the X values from the dataset
    Yvals[i] = arr[S[i]+1,11] # fill in Y values from dataset

#for i in range(0,n):
#    print(Xvals[i,:])
#    print(Yvals[i])

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

Xtest = np.zeros((len(arr)-TrainNum-1,12))
Ytest = np.zeros(len(arr)-TrainNum-1)
Xtest[:, :] = Xvals[TrainNum:len(arr), :]
Ytest[:] = Yvals[TrainNum:len(arr)]

# print(Xtest)
# print(Ytest)

#########################################################################
# Gradient Descent Method

numIterations = 200
alpha = 0.1
lambdas = [0,0.0005,0.001,0.0015,0.002,0.0025]
L1Error = np.zeros(len(lambdas))
L2Error = np.zeros(len(lambdas))
trainErr = np.zeros(len(lambdas))
m = len(Ytrain) # number of data points for training
arrCost = []; 
XT = np.transpose(Xtrain) 
theta = np.zeros(12)
atmp = 0.0

for k in range(0,len(lambdas)):
    print("Starting Training")
    print('Lambda: {}'.format(lambdas[k]))
    for iterations in range(0, numIterations): # Finding Theta
        sum = np.zeros(12) # initialize the sum to 0 for every iteration
        for i in range(0,m): 
            dot = theta.dot(XT[:,i]) 
            for j in range(0,12):
                sum[j] = (dot - Ytrain[i]) * XT[j,i] + sum[j] # Gradient Descent method
        sum[:] = alpha/m * sum[:]
        mul = 2*lambdas[k]*theta # second part of regularization
        theta = np.add(theta, mul) 
        theta = np.subtract(theta, sum)

        # Getting Cost Function 
        sum1 = 0
        for i in range(0,m):
            sum1 = (theta.dot(XT[:,i]) - Ytrain[i])**2 + sum1
        thetaT = np.transpose(theta)

        atmp = sum1/2/m + lambdas[k] * thetaT.dot(theta)
        #print(atmp)
        arrCost.append(atmp)

    trainErr[k] = atmp
    print('Training Error: {}'.format(trainErr[k]))
    print("Done training")

##########################################################################
# Test how well the predictions and ground-truth values match
    print("Start testing")
    testNum = len(Ytest)
    #print(testNum)

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
    L1Error[k] = L1Err/testNum
    L2Error[k] = L2Err/testNum
    print('L1 Error: {}'.format(L1Error[k]))
    print('L2 Error: {}'.format(L2Error[k]))
    print('Standard Deviation: {}'.format(STDErr))
    print("Done testing")

##########################################################################
# Plotting results of cost function from training

#    plt.plot(range(0,len(arrCost)),arrCost);
#    plt.xlabel('iteration')
#    plt.ylabel('cost')
#    plt.title('alpha = {} theta = {}'.format(alpha, theta))
#    plt.show()

plt.plot(range(0,len(lambdas)),L2Error,'r--',range(0,len(lambdas)),trainErr,'b--');
plt.xlabel('Lambda (x0.0005)')
plt.ylabel('L2 Error')
plt.title('Lambda vs. L2 Error')
plt.xticks()
plt.show()