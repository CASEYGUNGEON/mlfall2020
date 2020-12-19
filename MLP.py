import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

#load data
#wine_data = pd.read_csv('winequality-red.csv',sep = ';')
wine_data = pd.read_csv('winequality-white.csv',sep = ';')

# sns.countplot(data=wine_data, x='quality')
# plt.show()

# Data preprocessing
wine_data.loc[wine_data['quality'] <= 5, 'quality'] = 0
wine_data.loc[wine_data['quality'] >= 1, 'quality'] = 1

# sns.countplot(data=wine_data, x='quality')
# plt.show()

# Column
all_cols = wine_data.columns.tolist()

# feat column
feat_cols = all_cols[:-1]

# input
X = wine_data[feat_cols].values
# label
y = wine_data['quality'].values

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, random_state=10)

# Normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
mlp = MLPClassifier(hidden_layer_sizes=(60,60), activation='relu',max_iter=1000)
                #    solver='sgd', learning_rate='adaptive', learning_rate_init=0.1)
mlp.fit(X_train_scaled, y_train)
predict = mlp.predict(X_test_scaled)
accuracy = mlp.score(X_test_scaled, y_test)

print('accuracy is: {:.2f}%'.format(accuracy * 100))
print(' ')
print(sklearn.metrics.classification_report(y_test, predict))
print('\nconfusion matrix')
print(sklearn.metrics.confusion_matrix(y_test, predict))
    


