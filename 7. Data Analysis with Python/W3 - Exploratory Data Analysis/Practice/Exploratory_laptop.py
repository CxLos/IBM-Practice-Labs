import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# filepath = "./Data/laptop_wrangling2.csv"

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W3 - Exploratory Data Analysis\Data\laptop_wrangling2.csv'

df = pd.read_csv(file_path)

# print(df.head(5))
# print(df.dtypes)

# absolute_path = os.path.abspath(file_path)
# print(f"Absolute path of the file: {absolute_path}")

# 1. Visualize individual feature patterns. 

# Generate regression plots for each of the parameters below against the 'price' column. Print the value of Correlation of each feature with 'Price'.

# 1-1 CPU_frequency plot
# sns.regplot(x="CPU_frequency", y="Price", data=df)
# plt.ylim(0,)
# plt.show()

# 1-2 Screen_Size_inch plot
# sns.regplot(x="Screen_Size_in", y="Price", data=df)
# plt.ylim(0,)
# plt.show()

# 1-3 Weight_pounds plot
# sns.regplot(x="Weight_lbs", y="Price", data=df)
# plt.ylim(0,)
# plt.show()

# 1-4  All 3 previous values with Price
# for param in ['CPU_frequency', 'Screen_Size_in', 'Weight_lbs']:
#   print(f"Correlation of Price and {param} is ", df[[param,'Price']].corr())

# Box-plots with Categorical Features 'Category', 'GPU', 'OS', 'CPU_core', 'RAM_GB', 'Storage_GB_SSD'
  
# 1-5 Category Box-plot
# sns.boxplot(x="Category", y="Price", data=df)
# plt.show()
  
# 1-6 GPU Box-plot
# sns.boxplot(x="GPU", y="Price", data=df)
# plt.show()

# 1-7 OS Box-plot
# sns.boxplot(x="OS", y="Price", data=df)
# plt.show()

# 1-8 CPU_core Box-plot
# sns.boxplot(x="CPU_core", y="Price", data=df)
# plt.show()

# 1-9 RAM_GB Box-plot
# sns.boxplot(x="RAM_GB", y="Price", data=df)
# plt.show()

# 1-10 Storage_GB_SSD Box-plot
# sns.boxplot(x="Storage_GB_SSD", y="Price", data=df)
# plt.show()

# 2. DESCRIPTIVE STATISTICAL ANALYSIS

# Generate statistical description of all the features being used in the dataset. Include "object" data types.

describe = df.describe()
describe_obj = df.describe(include=['object'])
# print(describe)
# print(describe_obj)

# 3. GROUP BY & PIVOT TABLES

# Group  by the parameters "GPU", "CPU_core", and "Price" to make a pivot table and visualize this connection using teh pcolor plot

# 3-1. Create the group
group = df[['GPU','CPU_core','Price']]
grouped = group.groupby(['GPU','CPU_core'],as_index=False)['Price'].mean()
print(grouped)

# 3-2. Create the Pivot Table
pivot = grouped.pivot(index='GPU', columns='CPU_core')
pivot = pivot.fillna(0)
# print(pivot)

# 3-3. Create the plot
fig, ax = plt.subplots()
im = ax.pcolor(pivot, cmap='RdBu')

#label names
row_labels = pivot.columns.levels[1]
col_labels = pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)
# plt.show()


# 4. PEARSON CORRELATION & P-VALUES

# Use scipy.stats.pearsonr() function to evaluate the pearson coefficient and p-values for each parameter tested above.

pearson_coef, p_value = stats.pearsonr(df['GPU'], df['Price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

pearson_coef, p_value = stats.pearsonr(df['CPU_core'], df['Price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# df.to_csv('laptop-pricing-exploratory.csv')