# IMPORTS ----------------------------------------------------------------------------------------------------
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')  # optional: for ggplot-like style
# check for latest version of Matplotlib
print('Matplotlib version: ', mpl.__version__) # >= 2.0.0

# FETCHING DATA ----------------------------------------------------------------------------------------------

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

# Set Index
df_can.set_index('Country', inplace=True)

# Check dimensions
print('data dimensions:', df_can.shape)

# Variable to store the years
years = list(map(str, range(1980, 2014)))

# AREA PLOTS -------------------------------------------------------------------------------------------------

# Sort df in descending order
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

# get the top 5 entries
df_top5 = df_can.head()

# transpose the dataframe to invert the rows and columns
df_top5 = df_top5[years].transpose()

# HISTOGRAMS --------------------------------------------------------------------------------------------------

# np.histogram returns 2 values
count, bin_edges = np.histogram(df_can['2013'])

print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins

# BAR CHARTS --------------------------------------------------------------------------------------------------



# Horizontal Bar Chart (barh)


# PRINTS ------------------------------------------------------------------------------------------------------

print(df_can.head())

# df_can.to_excel('Canada_Immigration4.xlsx')