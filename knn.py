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

#read data from csv

data = pd.read_csv('.\\winequality-white.csv', sep = ';')
arr = pd.DataFrame(data).to_numpy()

