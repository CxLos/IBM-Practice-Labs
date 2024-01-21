# IMPORTS ----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt 
import numpy as np  # useful for many scientific computing in Python
import pandas as pd

# DATA ------------------------------------------------------------------------------------------------------
# .read_excel(path_to_excel_file, sheet_name_in_spreadsheet, skiprows= how many rows to skip, skip_footer= # of rows to skip at the bottom)

df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

# BASICS ---------------------------------------------------------------------------------------------------

# Remove a few columns
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

# Rename columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

# Add totals column
numeric_columns = df_can.select_dtypes(include=['int64', 'float64'])
df_can['Total'] = numeric_columns.sum(axis=1)
# print(df_can['Total'])

# INDEX & SELECTION (Slicing) -----------------------------------------------------------------------------

# Filter by Country
# print(df_can.Country[0:5])
# print(df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]][0:5])

# Select Row

# First change the index: 
df_can.set_index('Country', inplace=True)
# Resetting Index:
# df_can.reset_index()
# Remove name of the index:
# df_can.index.name = None

# Search specific country:
# print(df_can.loc['Japan'])
# print(df_can.loc['Dominican Republic'])

# Search by index
# print('Japan:', df_can.iloc[87])
# print(df_can[df_can.index == 'Japan'])
# print('Japan 2013 Value:', df_can.loc['Japan', 2013])

# year 2013 is the last column, with a positional index of 36, Japan index = 87
# print(df_can.iloc[87, 36])

# for years 1980 to 1985
# print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])

# 1980 - 1985 but with iloc
# print(df_can.iloc[87, [3, 4, 5, 6, 7, 8, 37]])

# View the number of immigrants from Haiti for the following scenarios:
# 1. The full row data (all columns)
# 2. For year 2000
# 3. For years 1990 to 1995

# print(df_can.loc['Haiti'])
# print(df_can[df_can.index == 'Haiti'])

# Displaying just the first 30 columns:
# .iloc[which rows, which columns]
# print(df_can[df_can.index == 'Haiti'].iloc[:, :30])

# print('Haiti Immigration 2000:', df_can.loc['Haiti', 2000])
# print(df_can[df_can.index == 'Haiti'][['Region',1990, 1991, 1992, 1993, 1994, 1995]])
# print(df_can.loc['Haiti', [1990, 1991, 1992, 1993, 1994, 1995]])

# Convert Years to strings
# First apply the map function to apply the str function to every column.
# map(function, iterable whose elements will be applied to the function)
# list(iterable). Basically turns any tuple, set, dictionary etc. and turns it into a list.
df_can.columns = list(map(str, df_can.columns))
# Iterate over every column and print what type the values are.
# [print (type(x)) for x in df_can.columns.values] 

# Declare a variable that will easily allow us to get the full range of years
years = list(map(str, range(1980, 2014)))
# print(years)

# Filtering Based on a criteria

# Filter on Asian countries only
condition = df_can['Continent'] == 'Asia'
# Pass a condition as a boolean vector to the dataframe:
asian_countries = df_can[condition]
# print('Asiasn Countries:', asian_countries.iloc[:, :27])

# Filter on Asian Countries and Region = Southern Asia
# df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]
condition2 = df_can['Region'] == 'Southern Asia'
se_asia = df_can[condition2]
# print(se_asia.iloc[:, :25])

# Sorting Values

# Sort df by 'Total' Column in descending order
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
top_5 = df_can.iloc[:10, [0,1,2,37]]
# print(top_5)

# QUESTIONS -----------------------------------------------------------------------------------------------

# 1. Fetch the data where area name is 'Africa' and 'RegName is 'Southern Africa'
condition3 = df_can['Region'] == 'Southern Africa'
s_africa = df_can[condition3]
# print(s_africa.iloc[:, :25])

# 2. Fetch data for top 3 countries where immigrants came from in 2010
df_can.sort_values(by='2010', ascending=False, axis=0, inplace=True)
top_3 = df_can.iloc[:10, [0,1,2,33]]
# print(top_3)

# PRINTS ---------------------------------------------------------------------------------------------------

# print(df_can.head())
# print(df_can.tail())
# print(df_can.columns)
# print(df_can.columns.tolist())
# print(df_can.index)
# print(df_can.index.tolist())
# print(type(df_can.columns))
# print(type(df_can.index))
# print(df_can.shape)
# print(df_can.describe())
# print(df_can.isnull().sum())

# Verbose = False provides more concise summary w/o non-null value counts
# print(df_can.info(verbose=False))

# Save to Excel File ---------------------------------------------------------------------------------------
df_can.to_excel('Canada_Immigration2.xlsx')

# MAP FUNCTION EXAMPLE ------------------------------------------------------------------------------------

# def square(x):
#     return x ** 2

# numbers = [1, 2, 3, 4, 5, 6 , 7, 8, 9, 10]
# squared_numbers = map(square, numbers)

# result_list = list(squared_numbers)

# print('Squared numbers:', result_list)

# ---------------------------------------------------------------------------------------------------------