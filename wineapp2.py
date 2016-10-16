# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 00:36:34 2016

@author: tom
"""

import pandas as pd
df = pd.read_csv('winequality-red.csv', sep = ';')
# Present the dataset
df.describe()

df.corr()

#Split the data into training and testing sets
from sklearn.cross_validation import train_test_split
Features = df[list(df.columns)[:-1]]
Quality = df['quality']
Features_train, Features_test, Quality_train, Quality_test = train_test_split(Features, Quality)

#Create and fit the model on the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Features_train, Quality_train)

#Evaluate the predictions of the model
Quality_predictions = regressor.predict(Features_test)


# Create scatterplot of Predicted Quality against True Quality 
import matplotlib.pylab as plt
plt.scatter(Quality_test, Quality_predictions)
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality ')
plt.show()
