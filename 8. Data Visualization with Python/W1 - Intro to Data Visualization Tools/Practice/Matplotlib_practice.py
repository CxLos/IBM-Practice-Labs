# Imports -----------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# DATASET -----------------------------------------------------------------------------------------------------------

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

# Set Index to 'Country'
df_can.set_index('Country', inplace=True)

# Optional: Remove name of index:
# df_can.index.name = None

# print(df_can.head())
print(df_can.dtypes)

# VISUALIZING DATA USING MATPLOTLIB ---------------------------------------------------------------------------------

# Check Matplotlib version:
# print('Matplotlib version: ', mpl.__version__)

# Get a list of all the different plotting styles that we can use in Matplotlib:
# print('List of available plotting styles:', plt.style.available)

# Apply a style to Matplotlib:
mpl.style.use(['ggplot']) 

# LINE PLOTS --------------------------------------------------------------------------------------------------------

# Create variable to easily call upon all the years in the dataset"
years = list(map(str, range(1980, 2014)))
# print('Years:', years)

# Create a dataseries for Haiti:
haiti = df_can.loc['Haiti', years]
# print(haiti.head())
# print(haiti.dtypes)
# print(type(haiti))

# Change index values(years) to int
# haiti.index = haiti.index.map(int) 

# Plot
haiti.plot(kind='line')

# # Add title and labels
plt.title('Immigration from Haiti')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
plt.text(2000, 6000, '2010 Earthquake')

# If years were type str:
# Year 2000 is the 20th index
plt.text(20, 6000, '2010 Earthquake')

# Display the plot
# plt.show()

# Dataseries for China & India:
df_CI = df_can.loc[['China', 'India'], years]

# Use transpose() method to swap the rows and columns
df_CI = df_CI.transpose()
# print(type(df_CI))

# Line Plot for CI
# df_CI.plot(kind='line')

# plt.title('Immigration from China and India')
# plt.xlabel('Year')
# plt.ylabel('Number of Immigrants')
# plt.show()

# Compare Immigration trends from top 5 countries
df_top5 = df_can.loc[['China', 'India', 'United Kingdom of Great Britain and Northern Ireland', 'Philippines', 'Pakistan'], years]
df_top5 = df_top5.transpose()
# print(df_top5)
# print(df_top5.dtypes)

df_top5.plot(kind='line')
plt.title('Immigration from Top 5 Countries')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
# plt.show()

# -------------------------------------------------------------------------------------------------------------------

# print(df_can.iloc[:5, :20])
# print(df_CI)
# print(df_CI.head())

# ------------------------------------------------------------------------------------------------------------------
# df_can.to_excel('Canada_Immigration3.xlsx')