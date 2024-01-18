# Importing Practice

import pandas as pd
import numpy as np
from pyodide.http import pyfetch

file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
file_name = "auto.csv"

#  Download the dataset

async def download(url, filename):
  response = await pyfetch(url)
  if response.status == 200:

    with open(filename, "wb") as f:
      f.write(await response.bytes())

  # Kind of connects the filename to the file path
  await download(file_path, "auto.csv")
  file_name = "auto.csv"

  df = pd.read_csv(file_name)

  print("The first 5 rows of the datafram:")
  print(df.head(5))

# download(file_path, file_name)

# Alternatively:
  
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
  
df = pd.read_csv(filepath, header=None)

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
# print("Headers:\n", headers)

df.columns = headers
df.columns

# Replace "?" with NaN.
df1 = df.replace('?',np.NaN)

# Drop missing values in the price column utilizing the pandas dropna() function
# subset = specifies which column the operation will be performed on
# axis = specifies if the operation should be performed on rows or columns. 0=rows, 1=columns.
df2 = df1.dropna(subset=["price"], axis=0)

# Save the datafram
# .to_csv saves the file in csv format
# index = specifies whether to write row names or not. true=yes, false=no.
df.to_csv("automobile.csv", index=False)

# Check Data types
print(df2.dtypes)

# Get Statistical summary of each column using the describe() method
# include = "all" means it will include all columns, even ones that are type object
# print(df2.describe())
# print(df2.describe(include = "all"))

# Use .info() method to get a concise summary of your dataframe
print(df.info())

# Print first 5 rows

# print("The first 5 rows of the datafram:")
# print(df2.head(5))

# 1. Check bottom 5 rows of dataframe

# print("The last 5 rows of the datafram:")
# print(df2.tail(5))

# 2. Find the name of the columns of the datafram

# print(df2.columns)

# 3. Apply the .describe() method to the columns: "length" and "comparison-ratio"
# print(df2[['length', 'compression-ratio']].describe())