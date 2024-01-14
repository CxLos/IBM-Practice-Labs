# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'

df = pd.read_csv(path)
# print(df.head())
# print(df['make'].unique())

# Calculate correlation between variables of type 'int64' or 'float64'
# Include only numeric columns in correlation calculation:
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()
# print(correlation_matrix)

# REGRESSION PLOTS

# Positive Linear Relationship Regression Plot between 'engine-size' and 'price'
# sns.regplot(x="engine-size", y="price", data=df)
# .ylim() is used to set the limits of the y-axis. the lower limit will be 0, and we will not set an upper limit.
# plt.ylim(0,)
# plt.show()

# Engine - Price correlation
enigine_price_correlation = df[["engine-size", "price"]].corr()
# print(enigine_price_correlation)

# Hwy-mpg - Price correlation
# sns.regplot(x="highway-mpg", y="price", data=df)
# plt.ylim(0,)
# plt.show()

hwympg_price_correlation = df[['highway-mpg', 'price']].corr()
# print(hwympg_price_correlation)

# Weak Linear Relationship Peak-rpm to price
# sns.regplot(x="peak-rpm", y="price", data=df)
# plt.ylim(0,48000)
# plt.show()

peakrpm_price_correlation = df[['peak-rpm','price']].corr()
# print(peakrpm_price_correlation)

# Box Plot Categorical variables
# sns.boxplot(x="body-style", y="price", data=df)
# plt.show()

# sns.boxplot(x="engine-location", y="price", data=df)
# plt.show()

# sns.boxplot(x="drive-wheels", y="price", data=df)
# plt.show()

# DESCRIPTIVE STATISTICAL ANALYSIS

# Run .describe() function
describe = df.describe()
# Including objects
# describe = df.describe(include=['object'])
# print(describe)

# Value_counts()
# Only works on Pandas series so we use single brackets
# wheels_counts = df['drive-wheels'].value_counts()

# .to_frame() to convert series to dataframe
wheels_counts = df['drive-wheels'].value_counts().to_frame()
# Rename column
wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# print(wheels_counts)

# Rename Index
wheels_counts.index.name = 'drive-wheels'
# print(wheels_counts)

# Value counts engine-location
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

# GROUPING BASICS

# Unique
# print(df['drive-wheels'].unique())

# Calculate avg price and then group by 'drive-wheels'
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False)['price'].mean()
# print(df_group_one)

# Group by multiple variables
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False)['price'].mean()
# print(grouped_test1)

# Pivot Table
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
# fill missing values with 0
grouped_pivot = grouped_pivot.fillna(0)
# print(grouped_pivot)

# Heat Map visualization of drive-wheels, body-style, and price
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()

# Fix the graph
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
# plt.show()

# CORRELATION AND CAUSATION

# Find Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Find Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

# Find Pearson Correlation Coefficient and P-value of 'length' and 'price'
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

# Find Pearson Correlation Coefficient and P-value of 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
# print("The Pearson Correlation Coefficient of width is", pearson_coef, " with a P-value of P =", p_value ) 

# Find Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
# print( "The Pearson Correlation Coefficient of curb-weight is", pearson_coef, " with a P-value of P = ", p_value) 

# Find Pearson Correlation Coefficient and P-value of 'engine-size' and 'price'
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
# print("The Pearson Correlation Coefficient of engine-size is", pearson_coef, " with a P-value of P =", p_value) 

# Find Pearson Correlation Coefficient and P-value of 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
# print("The Pearson Correlation Coefficient of bore is", pearson_coef, " with a P-value of P =  ", p_value ) 

# Find Pearson Correlation Coefficient and P-value of 'city-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
# print("The Pearson Correlation Coefficient of city-mpg is", pearson_coef, " with a P-value of P = ", p_value)  

# Find Pearson Correlation Coefficient and P-value of 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
# print("The Pearson Correlation Coefficient of highway-mpg is", pearson_coef, " with a P-value of P = ", p_value)  

# QUESTIONS

# 1. What is the data type of column 'peak-prm'?
# print(df.dtypes) 
# float64

# 2. find correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
# Double brackets ensures that we are selecting from a df with the specified columns, otherwise it would be interpreted as a pandas series.
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df[['bore','stroke','compression-ratio','horsepower']].corr()

# print(correlation_matrix)

# 3. Find correlation between x = 'stroke' and y = 'price'
# sns.regplot(x='stroke', y='price', data=df)
# plt.show()

stroke_price_correlation = df[['stroke','price']].corr()
# print(stroke_price_correlation)

# 4. Calculate avg price for each car based on body-style
df_group_two = df[['drive-wheels','body-style','price']]
df_group_two = df_group_two.groupby(['body-style'],as_index=False)['price'].mean()
# print(df_group_two)

# Save to .csv

# df.to_csv('cars.csv')