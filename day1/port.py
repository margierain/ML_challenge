import numpy as np
import pandas as pd
from sklearn.preprocessing import Impute,LabelEncoder, OneHotEncoder
from sklearn.cross_validatiom import train_test_split


day1_dataset = pd.read_csv('datasets/Data.csv')
#iloc(row,column)
X = day1_dataset.iloc(:, :-1) # get back all rows and columns excluding the last column
Y = day1_dataset.iloc(:, :3)
print(X, day1_dataset.iloc( ,'salary'))

imp_medain = Impute(missing_values='NAN', strategy='median', axix=0)
imp_medain = imp_medain.fit(X[:, 1:3])
[:, 1:3] = imp_medain.transform((X[:, 1:3])


labelEncoder = LabelEncoder()
print(X, '=======before encode')
X[:, 0] = labelEncoder.fit_transform(X[:, 0])
print(X)

oneHotEncoder_X = oneHotEncoder(categorrial_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray()
oneHotEncoder_Y = labelEncoder()
Y = oneHotEncoder_Y.fit_transform(Y)

X_trainset, X_testset, Y_trainset, Y_testset = train_test_split(X,Y, test_size=0.2, random_state=0)