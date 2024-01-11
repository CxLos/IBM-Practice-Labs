import numpy as np
import pandas as pd
import matplotlib as plt

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"

file_name="laptops.csv"

df = pd.read_csv(file_path, header=0)

# Round screen size to 2 decimals
df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)

# 1. Identify which columns have missing values
missing_data = df.isnull()

# for column in missing_data.columns:
    # print(column)
    # print (missing_data[column].value_counts())
    # print("")   

# 2. Replace the missing values in the "Weight_kg" column with the mean. Replace missing values of screen_size with the mode.
    
avg_weight = df["Weight_kg"].astype('float').mean(axis=0)
df['Weight_kg'].replace(np.nan, avg_weight, inplace=True)

screen_mode = df['Screen_Size_cm'].astype('float').value_counts().idxmax()
df['Screen_Size_cm'].replace(np.nan, screen_mode, inplace=True)
# print(screen_mode)

# 3. Convert columns "Weight_kg" and "Screen_Size_cm" to float and rename
    
df[['Weight_kg', 'Screen_Size_cm']] = df[['Weight_kg', 'Screen_Size_cm']].astype('float')

# 4. Modify screen_size column to inches, and weight_kg to pounds. Normalize the column "CPU_frequency"
# 1in = 2.54cm
# 1kg = 2.205 lbs
    
df['Screen_Size_cm'] = df['Screen_Size_cm']/2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_in'}, inplace=True)
df[['Screen_Size_in']] = np.round(df[['Screen_Size_in']],3)

df['Weight_kg'] = df['Weight_kg']*2.205
df.rename(columns={'Weight_kg':'Weight_lbs'}, inplace=True)
df[['Weight_lbs']] = np.round(df[['Weight_lbs']],2)

df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()

# 5. Create 3 bins of low, medium, high for the "price" column and name it "Price-binned". Graph the results
    
bins = np.linspace(min(df['Price']), max(df['Price']), 4)
group_names = ['Low', 'Medium', 'High']
df['Price-binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest=True)

# 6. Create dummy variables for the "Screen" column and rename them "Screen-IPS_panel" and "Screen_Full-HD" then drop the "Screen" column

dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen_IPS-Panel', 'Full HD':'Screen_Full-HD'}, inplace=True)
df.drop("Screen", axis=1, inplace=True)

df = pd.concat([df, dummy_variable_1], axis=1)


print(df.head())
# print(df.info())
# print(df.describe())
# print(df["Manufacturer"].unique())
# print(dummy_variable_1)

# df.to_csv('laptop_wrangling2.csv')