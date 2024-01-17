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

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W5 - Model Evaluation & Refinement\Data\laptop_model_evaluation.csv'

df = pd.read_csv(file_path)
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

# TASK 1: CROSS-VALIDATION -------------------------------------------------------------------------------

# 1-1. Divide the dataset into x_data and y_data parameters. Here y_data is the "Price" attribute, and x_data has all other attributes in the data set.


# 1-2. Split the data set into training and testing subests such that you reserve 10% of the data set for testing purposes.


# 1-3. Create a single variable linear regression model using "CPU_frequency" parameter. Print the R^2 value of this model for the training and testing subsets.


# 1-4. Run a 4-fold cross validation on the model and print the mean value of R^2 score along with its standard deviation.


# TASK 2: OVERFITTING ------------------------------------------------------------------------------------

# 2-1.


# 2-2. 


# 2-3. 


# TASK 3: RIDGE REGRESSION -------------------------------------------------------------------------------

# 3-1. 


# 3-2. 


# 3-3. 


# TASK 4: GRID SEARCH ------------------------------------------------------------------------------------

print(df.head())