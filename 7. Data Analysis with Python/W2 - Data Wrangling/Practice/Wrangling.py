import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import pyplot
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

    await download(file_path, "usedcars.csv")
    file_name="usedcars.csv"

file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

file_name="usedcars.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(file_path, names = headers)

# CLEANING THE DATA

# replace "?" to NaN
# inplace=True means the modifications will be directly applied to the existing datafram
df.replace("?", np.nan, inplace = True)

# Evaluate for missing data
missing_data = df.isnull()
# print(missing_data.head(5))

# Count Missing Data
# for column in missing_data.columns.values.tolist():
    # print(column)
    # print (missing_data[column].value_counts())
    # print("")    

# Calculate Average for the Normalized-Losses column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
# print("Average of normalized-losses:", avg_norm_loss)

# Replace "NaN" with mean value in the Normalized-Losses column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Calculate mean value for "bore" column
avg_bore=df['bore'].astype('float').mean(axis=0)
# print("Average of bore:", avg_bore)

# Replace "NaN" with mean value in the bore column
df["bore"].replace(np.nan, avg_bore, inplace=True)

# Calculate mean Horsepower and subbing it for NaN
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
# print("Average horsepower:", avg_horsepower)

# Calculate mean peak-rpm and subbing it for Nan
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['num-of-doors'].value_counts()
# print("Average peak rpm:", avg_peakrpm)

# See which values are present in a particular column
# .value_counts() returns a count of all the unique values in a column
df["num-of-doors"].replace(np.nan, "four", inplace=True)
# print(df['num-of-doors'].value_counts())
# Check for most common type:
# .idxmax() returns the first occurrence of max value
# print(df['num-of-doors'].value_counts().idxmax())

# Drop all rows that do not have data in price column
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# CORRECTING DATA FORMAT

# copy=True creates copy of df, copy=false will modify the original object
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df["horsepower"] = df["horsepower"].astype(int, copy=True)

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# DATA NORMALIZATION

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
# print(df[['length','width','height']].head(10))

# BINNING

# Create bin by setting start_value, end_value, and numbers_generated
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
# print("Bins:", bins)

# Set group names
group_names = ['Low', 'Medium', 'High']

# Applying .cut() method
# .cut(input_array, bins, labels, right, include_lowest, precision)
# right=True by default, meaning that the right end-point is included in the interval
# include_lowest specifies whether the first interval should be left-closed or not, meaning that the left end-point will be included in the interval (0)
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
# print(df[['horsepower','horsepower-binned']].head(20))

# See # of vehicles in each bin
# print(df["horsepower-binned"].value_counts())

# PLOTTING DATA

# .bar(x-coordinate, height, width, bottom, align)
# bar_graph = plt.bar(group_names, df["horsepower-binned"].value_counts())
histogram = plt.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
# plt.show()

# INDICATOR VARIABLES

# Get dummies
df.columns
dummy_variable_1 = pd.get_dummies(df["fuel-type"])

# Change column names
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
# print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# PRACTICE PROBLEMS

# 1. In stroke column, replace NaN with mean value
avg_stroke = df['stroke'].astype('float').mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
# print("Average Stroke:",avg_stroke)

# 2. Transform mpg to L/100km in the column "highway-mpg" and change the name of the column to "highway-L/100km"
df['highway-L/100km'] = 235/df['highway-mpg']

# 3. Normalize height column
df['height'] = df['height']/df['height'].max()

# 4. Create indicator/dummy variable for "aspiration" column
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
# print(dummy_variable_2.head())

# 5. Merge all the dataframes and drop the "aspiration" column
df = pd.concat([df, dummy_variable_1, dummy_variable_2], axis=1)
df.drop("aspiration", axis=1, inplace=True)

# print(df.head(10))
# print(df.dtypes)
# print(df['horsepower'].min)

# Save Data to .csv file
# df.to_csv('imports_df.csv')