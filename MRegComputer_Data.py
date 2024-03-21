# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:14:46 2024

@author: HP
"""

#Problem statement
'''
	Perform multilinear regression with price as the output variable and 
    document the different RMSE values.
'''
#Business Objective
'''
    The business objective is not explicitly defined in the provided information. 
    It appears that the task is to perform multilinear regression with 'price' 
    as the output variable
'''
# Data Dictionary
'''
| Column Name | Data Type | Description              | Relevant to Model |
|-------------|-----------|--------------------------|-------------------|
| Unnamed: 0  | Numeric   | Index or identifier      | No                |
| price       | Numeric   | Price of the computer    | Yes               |
| speed       | Numeric   | Processor speed          | Yes               |
| hd          | Numeric   | Hard drive capacity      | Yes               |
| ram         | Numeric   | RAM capacity             | Yes               |
| screen      | Numeric   | Screen size              | Yes               |
| ads         | Numeric   | Advertising budget       | Yes               |
| trend       | Numeric   | Trend in technology      | Yes               |
'''

#Notes:
'''    
    - 'Unnamed: 0' is an index or identifier column and is not relevant to model building.
    - All other columns contain numerical data and are relevant to model building.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('C:/0-Assignments/Assignments/MultiReg/Computer_Data.csv.xls')

# Data Pre-processing
# Drop non-numeric columns or handle them appropriately
data_numeric = data.select_dtypes(include=[np.number])

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

# Exploratory Data Analysis (EDA)
# Summary Statistics
summary_statistics = data_numeric.describe()
print("Summary Statistics:\n", summary_statistics)

# Univariate Analysis
plt.figure(figsize=(15, 10))
for i, column in enumerate(data_numeric.columns):
    plt.subplot(rows, 3, i+1)
    sns.histplot(data[column], kde=True)
    plt.title(f"{column} Distribution")
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(15, 10))
for i, column in enumerate(data_numeric.columns[1:]):
    plt.subplot(rows, 3, i+1)
    sns.scatterplot(data=data_numeric, x=column, y='price')
    plt.title(f"{column} vs Price")
plt.tight_layout()
plt.show()

# Model Building
# Prepare data
X = data_numeric.drop(columns=['Unnamed: 0', 'price'])
y = data_numeric['price']

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
print("Training RMSE:", train_rmse) #313.0874184482436
print("Training R-squared:", train_r2)  #0.7115753439192618
print("Test RMSE:", test_rmse)  #305.4279203576135
print("Test R-squared:", test_r2)   #0.7144365518566644

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test['speed'], y_test, color='blue', label='Actual')
plt.plot(X_test['speed'], y_test_pred, color='red', label='Predicted')
plt.title('Regression Line')
plt.xlabel('Processor Speed')
plt.ylabel('Price')
plt.legend()
plt.show()

