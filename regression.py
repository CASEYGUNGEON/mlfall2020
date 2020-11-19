# Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# get min
minval = np.zeros(13)
dif = np.zeros(13)
for i in range(0,12):
    minval[i] = arr[:,i].min()
    dif[i] = arr[:,i].max() - minval[i]
    for j in range(0,4898):
        arr[j,i] = (arr[j,i] - minval[i]) / dif[i]

print(arr)

# Assign values for X and Y
Xvals = np.ones((4898,12))
Yvals = np.zeros(4898)
Xvals[:, 1:12] = arr[:, 0:11]
Yvals[:] = arr[:, 11]

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
[a,b] = XT.shape
print(a)
print(b)
# Finding Theta
for iterations in range(0, numIterations):
    sum = np.zeros(12)
    #print(iterations)
    for i in range(0,m):
        dot = theta.dot(XT[:,i])
        for j in range(0,12):
            sum[j] = (dot - Ytrain[i]) * XT[j,i] + sum[j]
            #print(XT[j,i])
    sum[:] = alpha/(m+1) * sum[:]
    theta = np.subtract(theta, sum)

# Getting Cost Function 

    sum1 = 0
    for i in range(0,m):
        sum1 = (theta.dot(XT[:,i]) - Ytrain[i])**2 + sum1
    atmp = sum1/2/(m+1)
    #print(atmp)
    arrCost.append(atmp)
print("Done training")

##########################################################################
# Plotting results of cost function

plt. plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {} theta = {}'.format(alpha, theta))
plt.show()