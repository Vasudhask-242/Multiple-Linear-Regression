# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:02:17 2024

@author: HP
"""

#Problem Statement
'''
    With the growing consumption of avocados in the USA, a freelance company 
    would like to do some analysis on the patterns of consumption in different 
    cities and would like to come up with a prediction model for the price of 
    avocados. For this to be implemented, build a prediction model using 
    multilinear regression and provide your insights on it.
'''
# Business Objective
'''
    The freelance company aims to analyze the consumption patterns of avocados in 
    different cities in the USA and develop a prediction model for avocado prices. 
    By understanding consumption trends and price dynamics, the company seeks to 
    provide insights to avocado producers, retailers, and consumers, facilitating 
    informed decision-making and improving market efficiency.
'''
# Data Dictionary
'''
    | Column Name   | Data Type | Description                              
    |---------------|-----------|------------------------------------------
    | AveragePrice  | Numeric   | Average price of avocados                
    | Total_Volume  | Numeric   | Total volume of avocados sold           
    | tot_ava1      | Numeric   | Total volume of avocados of type 1       
    | tot_ava2      | Numeric   | Total volume of avocados of type 2       
    | tot_ava3      | Numeric   | Total volume of avocados of type 3       
    | Total_Bags    | Numeric   | Total number of bags                      
    | Small_Bags    | Numeric   | Number of small bags                      
    | Large_Bags    | Numeric   | Number of large bags                      
    | XLarge_Bags   | Numeric   | Number of extra large bags                
    | year          | Numeric   | Year                                     
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('C:/0-Assignments/Assignments/MultiReg/Avacado_Price.csv.xls')
data.shape
#(18249, 12)

# Drop non-numeric columns or handle them appropriately
data_numeric = data.select_dtypes(include=[np.number])
data_numeric.shape
#(18249, 10)

# Summary Statistics
summary_statistics = data_numeric.describe()

# Data Pre-processing
# Check for missing values
missing_values = data_numeric.isnull().sum()
print("Missing Values:\n", missing_values)


# Outlier Treatment
# Visualize numerical features using boxplots
plt.figure(figsize=(15, 10))
num_cols = len(data_numeric.columns)
rows = (num_cols - 1) // 3 + 1  # Calculate number of rows for subplots
for i, column in enumerate(data_numeric.columns):
    plt.subplot(rows, 3, i+1)
    sns.boxplot(data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()


# Model Building
# Prepare data
X = data_numeric.drop(columns=['AveragePrice'])
y = data_numeric['AveragePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multilinear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
# Training set
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Test set
y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("Multilinear Regression Results:")
print("Training RMSE:", train_rmse) #0.39059352331172287
print("Training R-squared:", train_r2)  #0.061154139616380165
print("Test RMSE:", test_rmse)  #0.3899704368895652
print("Test R-squared:", test_r2) #0.05347725581322815

# Plot regression lines for each feature against the target variable

plt.figure(figsize=(15, 10))
for i, column in enumerate(X_train.columns):
    plt.subplot(3, 3, i+1)
    
    # Scatter plot of actual data points
    plt.scatter(X_train[column], y_train, color='blue', label='Actual')
    
    # Predicted values
    plt.plot(X_train[column], y_train_pred, color='red', label='Predicted')
    
    plt.title(f"Regression Line for {column} vs AveragePrice")
    plt.xlabel(column)
    plt.ylabel('AveragePrice')
    plt.legend()

plt.tight_layout()
plt.show()

