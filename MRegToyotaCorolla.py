# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:43:16 2024

@author:HP
"""

#Problem Statement
'''
    An online car sales platform would like to improve its customer base and 
    their experience by providing them an easy way to buy and sell cars. 
    For this, they would like an automated model which can predict the price of
    the car once the user inputs the required factors. Help the business achieve 
    their objective by applying multilinear regression on the given dataset. 
    Please use the below columns for the analysis purpose: 
    price, age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, and Weight.

'''
# Business Objective
'''
    The business objective is to create an automated model that predicts the 
    price of cars on an online sales platform. By providing users with accurate 
    price predictions based on relevant factors, the platform aims to enhance customer 
    experience and facilitate buying and selling transactions.
'''

# Data Dictionary
'''
| Column Name   | Data Type | Description                              | Relevant to Model |
|---------------|-----------|------------------------------------------|-------------------|
| Price         | Numeric   | Price of the car (target variable)       | Yes               |
| age_08_04     | Numeric   | Age of the car in months                 | Yes               |
| KM            | Numeric   | Distance covered in kilometers           | Yes               |
| HP            | Numeric   | Horsepower                               | Yes               |
| cc            | Numeric   | Cylinder capacity in cubic centimeters   | Yes               |
| Doors         | Numeric   | Number of doors                          | Yes               |
| Gears         | Numeric   | Number of gears                          | Yes               |
| Quarterly_Tax | Numeric   | Quarterly road tax                       | Yes               |
| Weight        | Numeric   | Weight of the car in kilograms           | Yes               |
'''
# Regression Line Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('C:/0-Assignments/Assignments/MultiReg/ToyotaCorolla.csv.xls',encoding='ISO-8859-1')
data.shape
#(1436, 38)


# Select relevant columns
selected_columns = ['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']
data = data[selected_columns]

# Data Pre-processing
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Outlier Treatment
# Visualize numerical features using boxplots
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()

# Exploratory Data Analysis (EDA)
# Summary Statistics
summary_statistics = data.describe()
print("Summary Statistics:\n", summary_statistics)
'''
Summary Statistics:
               Price    Age_08_04  ...  Quarterly_Tax      Weight
count   1436.000000  1436.000000  ...    1436.000000  1436.00000
mean   10730.824513    55.947075  ...      87.122563  1072.45961
std     3626.964585    18.599988  ...      41.128611    52.64112
min     4350.000000     1.000000  ...      19.000000  1000.00000
25%     8450.000000    44.000000  ...      69.000000  1040.00000
50%     9900.000000    61.000000  ...      85.000000  1070.00000
75%    11950.000000    70.000000  ...      85.000000  1085.00000
max    32500.000000    80.000000  ...     283.000000  1615.00000

[8 rows x 9 columns]
'''

# Univariate Analysis
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[column], kde=True)
    plt.title(f"{column} Distribution")
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns[1:]):  # Exclude 'Price' from bivariate analysis
    plt.subplot(3, 3, i+1)
    sns.scatterplot(data=data, x=column, y='Price')
    plt.title(f"{column} vs Price")
plt.tight_layout()
plt.show()

# Model Building
# Prepare data
X = data.drop(columns=['Price'])
y = data['Price']

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
print("Training RMSE:", train_rmse) #1327.1273835742518
print("Training R-squared:", train_r2)  #0.865485386905346
print("Test RMSE:", test_rmse)  #1396.5117213222475
print("Test R-squared:", test_r2)   #0.8538352805672252

# Plot regression line
plt.figure(figsize=(10, 6))

# Scatter plot of actual data points
plt.scatter(X_train['Age_08_04'], y_train, color='blue', label='Actual')

# Predicted values
plt.plot(X_train['Age_08_04'], y_train_pred, color='red', label='Predicted')

plt.title('Regression Line for Age vs Price')
plt.xlabel('Age of the Car (Months)')
plt.ylabel('Price')
plt.legend()
plt.show()
