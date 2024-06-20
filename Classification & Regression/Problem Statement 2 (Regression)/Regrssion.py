# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:44:01 2024

@author: Win10
"""

import pandas as pd
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# reading the dataset
data = pd.read_csv("D:\Projects\Spyder\Assig1(ML)\Problem Statement 2 (Regression)\California_Houses.csv")

print(data)
# Some Info About The Dataset
print("==" * 40)
print("Data Info")
print(data.info())
print(data.describe())
print("==" * 40)

# Check If There Is Null Values
print("==" * 40)
print("Check For Null Values")

print(data.isnull().sum())  # No Need For Removing The NULL (data = data.dropna())
print(f"Number Of Null Values: {data.isnull().sum().sum()}")
print("==" * 40)

# Check If There Duplicated Values
print("==" * 40)
print("Check For Duplicated Values")
print(data.duplicated())
print(data.duplicated().sum())  # No Need For Removing The Duplicates (data = data.drop_duplicates(inplace=True))
print("==" * 40)

# Linear Regression
# [1] Splitting The Data (70% Training , 15% Validation , 15% Testing)

# data = pd.get_dummies(data)
print("==" * 40)
x_labels = data[[
            'Median_Income',
            'Median_Age',
            'Tot_Rooms',
            'Tot_Bedrooms',
            'Population',
            'Households',
            'Latitude',
            'Longitude',
            'Distance_to_coast',
            'Distance_to_LA',
            'Distance_to_SanDiego',
            'Distance_to_SanJose',
            'Distance_to_SanFrancisco'
            ]]

y_label = data['Median_House_Value']

# Split the data
"""
    x_train and y_train For Learn The Model 
    x_test and y_test For Test The Performance 
    x_val and y_val For Tune The Model Parameters 
"""
x_train, x_test, y_train, y_test = train_test_split(x_labels, y_label, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Print The Number Of Col And Row For Each Dataset
print("Training set:", x_train.shape, y_train.shape)
print("Validation set:", x_val.shape, y_val.shape)
print("Test set:", x_test.shape, y_test.shape)

# Build The Model
# Step1: Train The Model By Training Dataset
model = LinearRegression().fit(x_train , y_train)

# Step2: Give The Model Validation Dataset To Tune The Model Parameters
y_val_predict = model.predict(x_val)

print(y_val_predict) # ==> Result

# Evaluate The Result
mse_val = mean_squared_error(y_val , y_val_predict); # mean_squared_error(ground_truth , predicted_values)
print(f"Mean Square Error(Validation): {mse_val}")


# Step 3: Test The Model
y_test_predict = model.predict(x_test)

print("Score: " , model.score(x_test , y_test)) # ==> Result

# Evaluate The Result
mse_test = mean_squared_error(y_test , y_test_predict)
print(f"Mean Square Error(Test): {mse_test}")

print("==" * 40)


# Check
# Let's Take One Random Datapoint
# Record Number 11546
# x_labels Values
# 7.0879	16	2070	263	878	297	33.76	-118.04	5632.876204	37540.12471	142001.5529	528645.1146	596654.4088
# Y = 338800



dic = {
    'Median_Income': 7.0879,
    'Median_Age': 16,
    'Tot_Rooms': 2070,
    'Tot_Bedrooms': 263,
    'Population': 878,
    'Households': 297,
    'Latitude': 33.76,
    'Longitude': -118.04,
    'Distance_to_coast': 5632.876204,
    'Distance_to_LA': 37540.12471,
    'Distance_to_SanDiego': 142001.5529,
    'Distance_to_SanJose': 528645.1146,
    'Distance_to_SanFrancisco': 528645.1146
}

df = pd.DataFrame([dic])


y = model.predict(df)

print("Result (Linear Regression): " , y) # ==> 340745.43860487

# loss(y , f(x)) = abs(338800 - 340745.43860487) = 1945.4386048699962

print("==" * 40)
print("Lasso Regression")
# Lasso Regression

# Step1 : Learn The Model
model = Lasso(max_iter=5000).fit(x_train , y_train)

# Step2: Give The Model Validation Dataset To Tune The Model Parameters
y_val_predict = model.predict(x_val)

print(y_val_predict) # ==> Result

# Evaluate The Result
mse_val = mean_squared_error(y_val , y_val_predict); # mean_squared_error(ground_truth , predicted_values)
print(f"Mean Square Error(Validation): {mse_val}")


# Step 3: Test The Model
y_test_predict = model.predict(x_test)

print(y_test_predict) # ==> Result

# Evaluate The Result
mse_test = mean_squared_error(y_test , y_test_predict)
print(f"Mean Square Error(Test): {mse_test}")

print("==" * 40)

# Check
df = pd.DataFrame([dic])


y = model.predict(df)

print("Result(Lasso Regression): " , y) # ==> 340744.55951878
# loss(y , f(x)) = abs(338800 - 340744.55951878) = 1944.5595187799772

print("==" * 40)
print("Ridge Regression")
# Ridge Regression

# Step1 : Learn The Model
model = Ridge().fit(x_train , y_train)

# Step2: Give The Model Validation Dataset To Tune The Model Parameters
y_val_predict = model.predict(x_val)

print(y_val_predict) # ==> Result

# Evaluate The Result
mse_val = mean_squared_error(y_val , y_val_predict); # mean_squared_error(ground_truth , predicted_values)
print(f"Mean Square Error(Validation): {mse_val}")


# Step 3: Test The Model
y_test_predict = model.predict(x_test)

print(y_test_predict) # ==> Result

# Evaluate The Result
mse_test = mean_squared_error(y_test , y_test_predict)
print(f"Mean Square Error(Test): {mse_test}")

print("==" * 40)

# Check
df = pd.DataFrame([dic])


y = model.predict(df)

print("Result (Ridge Regression): " , y) # ==> 340741.13551039
# loss(y , f(x)) = abs(338800 - 340741.13551039) = 1941.1355103899841


# Linear Regression ==> loss => 1945.4386048699962 [3]
# Lasso Regression ==> loss => 1944.5595187799772 [2]
# Ridge Regression ==> loss => 1941.1355103899841 [1] (Best One) (Least Error)