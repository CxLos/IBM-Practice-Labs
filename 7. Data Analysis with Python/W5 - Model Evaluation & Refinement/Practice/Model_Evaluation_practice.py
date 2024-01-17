# Import
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Data
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W5 - Model Evaluation & Refinement\Data\auto5.csv'

df = pd.read_csv(file_path)
print(df.head())
# print(df.dtypes)

# PT. 1 TRAINING & TESTING -----------------------------------------------------------



# PT.2 OVERFITTING, UNDERFITTING & MODEL SELECTION ----------------------------------



# PT. 3 RIDGE REGRESSION -----------------------------------------------------------



# PT. 4 GRID SEARCH -----------------------------------------------------------------


# QUESTIONS

# 1. Use tain_test_split() to split up dataset so that 40% of the data samples will be utilized for testing. Set parameter "random_state" = 0.


# 2. Find the R^2 on the test data using 40% of the dataset for testing.



# 3. Calculate the avg R^2 using 2 folds, then find the avg R^2 for the second fold Utilizing the "horsepower" feature.



# 4a. Create a "PolynomialFeatures" obj "pr1" of degree = 2.


# 4b. Transform the training and testing samples for the features "horsepower", "curb-weight", "engine-size" and "highway-mpg" utilizing the "fit_transform" method.


# 4c. How many dimensions does teh new feature have? Use "shape" attribute.


# 4d. Create a Linear Regression Model "poly1" and then train the obj using the fit method using the polynomial features.


# 4e. Use the predict method to predict an output on the polynomial features, then use the "DistributionPlot()" function to display the distribution of the predicted test output vs. the actual test data.


# 4f. Using previous distribution plot, descirbe the two regions where the predicted prices are less accurate than the actual prices.


# 5. Perform a Ridge Regression. Calculate the R^2 using the polynomial features, use the training data to train the model and use the test data to test the model with alpha parameter = 10.


# 6. Perform a Grid Search for the alpha parameter and the normalization parameter, then find the best values of the parameters.


# df.to_csv('auto5.csv')