#KNN model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#read data from csv

data = pd.read_csv('.\\winequality-white.csv', sep = ';')
arr = pd.DataFrame(data).to_numpy()