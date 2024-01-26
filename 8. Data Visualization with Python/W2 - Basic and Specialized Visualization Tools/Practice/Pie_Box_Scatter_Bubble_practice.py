# IMPORT LIBRARIES  --------------------------------------------------------------------------------------------

import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

# optional: for ggplot-like style
mpl.style.use('ggplot') 

# IMPORT DATA --------------------------------------------------------------------------------------------------

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

# VISUALIZING DATA USING MATPLOTLIB ---------------------------------------------------------------------------

# Set Index
df_can.set_index('Country', inplace=True)

# Store the years in a variable
years = list(map(str, range(1980, 2014)))

# group countries by continents and apply sum() function 
df_continents = df_can.groupby('Continent').sum()

# print(df_continents.iloc[:, 36])
# print(print(type(df_can.groupby('Continent', axis=0))))

# PIE CHARTS -------------------------------------------------------------------------------------------------

# View Immigration by continent

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

# autopct create %, start angle represent starting point
# df_continents['Total'].plot(kind='pie',
#                             figsize=(10, 6),
#                             autopct='%1.1f%%', # add in percentages. 1 digit to the left of the decimal, & 1 to the right.
#                             startangle=90,     # Rotate the start of the pie. start angle 90° (Africa)
#                             shadow=True,       # add shadow      
#                             labels=None,         # turn off labels on pie chart
#                             pctdistance=1.11,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
#                             colors=colors_list,  # add custom colors
#                             explode=explode_list # 'explode' lowest 3 continents
#                             )

# # scale the title up by 12% to match pctdistance
# plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.085, fontsize = 15) 
# plt.axis('equal') # Sets the pie chart to look like a circle.
# plt.legend(labels=df_continents.index, loc='upper left') 
# plt.show()

# Immigration to Canada in 2013 by continent

# df_continents['2013'].plot(kind='pie',
#                             figsize=(10, 6),
#                             autopct='%1.1f%%', # add in percentages. 1 digit to the left of the decimal, & 1 to the right.
#                             startangle=90,     # Rotate the start of the pie. start angle 90° (Africa)
#                             shadow=True,       # add shadow      
#                             labels=None,         # turn off labels on pie chart
#                             pctdistance=1.11,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
#                             # colors=colors_list,  # add custom colors
#                             # explode=explode_list # 'explode' lowest 3 continents
#                             )

# # scale the title up by 12% to match pctdistance
# plt.title('Immigration to Canada by Continent 2013', y=1.085, fontsize = 15) 
# plt.axis('equal') # Sets the pie chart to look like a circle.
# plt.legend(labels=df_continents.index, loc='upper left') 
# plt.show()

# BOX PLOTS -------------------------------------------------------------------------------------------------

df_japan = df_can.loc[['Japan'], years].transpose()
# print(df_japan.head())
# print(df_japan.describe())

# df_japan.plot(kind='box', figsize=(8, 6))

# plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
# plt.ylabel('Number of Immigrants')
# plt.show()

# Compare Distribution of the number of new immigrants from India & China from 1980-2013

df_CI = df_can.loc[['India', 'China'], years].transpose()
# print(df_CI)
# print(df_CI.describe())

# Vertical
# df_CI.plot(kind='box', figsize=(8, 6))

# Horizontal
# df_CI.plot(kind='box', figsize=(10, 7), color='blue', vert=False)

# plt.title('Box plot of Immigrants from India & China 1980 - 2013')
# plt.ylabel('Number of Immigrants')
# plt.show()

# SUBPLOTS

# fig = plt.figure() # create figure

# ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
# ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot)

# Alternatively if all inputs are less than 10, it can be written like:
# subplot(211) == subplot(2,1,1)

# Subplot 1: Box plot
# df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
# ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
# ax0.set_xlabel('Number of Immigrants')
# ax0.set_ylabel('Countries')

# # Subplot 2: Line plot
# df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
# ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
# ax1.set_ylabel('Number of Immigrants')
# ax1.set_xlabel('Years')

# plt.show()

# Create a Box Plot to Visualize distribution of top 15 countries grouped by decade

# Varoable for top 15 countries
df_decade = df_can.sort_values(['Total'], ascending=False, axis=0).head(15)

# Variables to store years by decade
year8 = list(map(str, range(1980, 1989)))
year9 = list(map(str, range(1990, 1999)))
year2 = list(map(str, range(2000, 2009)))

# Create mini dataframes for each decade
df_8 = df_decade.loc[:, year8].sum(axis=1)
df_9 = df_decade.loc[:, year9].sum(axis=1)
df_2 = df_decade.loc[:, year2].sum(axis=1)
# print(df_2)

# New Dataframe
new_df = pd.DataFrame({'1980s': df_8, '1990s': df_9, '2000s': df_2})

# print(new_df.head())
# print('Quartile Ranges: \n', new_df.describe().loc[['25%', '50%', '75%']])

# Plot it

# Horizontal
# new_df.plot(kind='box', figsize=(10, 7), color='blue', vert=False)

# plt.title('Box plot of Immigrants from top 15 Countries by decade')
# plt.ylabel('Decade')
# plt.show()

# Outliers

# Q1 = 32,037
# Q3 = 97,459
# IQR = 65,422
# Above Outlier = 195,592
# print('Interquartile Range:', 97459 - 32037)
# print('Above outlier threshold:', 97459 + (1.5 * 65422))

# How many outliers in our data

new_df = new_df.reset_index()
# print(new_df[new_df['2000s'] > 195592])

# SCATTER PLOTS ---------------------------------------------------------------------------------------------

# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0)) # axis=0 to calculate sum for each column

# change the years to type int (useful for regression later on)
df_tot.index = map(int, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
# print(df_tot.head())

# Plot

# df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
# x = df_tot['year']      # year on x-axis
# y = df_tot['total']     # total on y-axis
# fit = np.polyfit(x, y, deg=1) # deg=1 equals linear fit (straight line)

# # plot line of best fit
# plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years

# # y=mx+b. format() places values of actual variable into the equation below.
# # xy = coordinates where we want to annotate the plot
# # .0f = formats coeffiecients as floating-point numbers w/ no decimal
# plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

# plt.title('Total Immigration to Canada from 1980 - 2013')
# plt.xlabel('Year')
# plt.ylabel('Number of Immigrants')
# plt.show()

# print('No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) )

# Using the equation of line of best fit, we can estimate the number of immigrants in 2015:

# print('No. Immigrants:', 5567 * years[20] - 10926195)
# print('No. Immigrants =', 5567 * 2015 - 10926195)
# print('No. Immigrants =', 291,310)

# Scatter Plot of total immigration to Canada from Denmark, Norway & Sweden from 1980-2013

df_scan = df_can.loc[['Norway', 'Denmark', 'Sweden'], years].transpose()
# print(df_scan)

# create df_total by summing across three countries for each year
df_total = pd.DataFrame(df_scan.sum(axis=1)) # axis=1 for rows

# reset index in place
df_total.reset_index(inplace=True)

# rename columns
df_total.columns = ['year', 'total']

# change column year from string to int to create scatter plot
df_total['year'] = df_total['year'].astype(int)

# print(df_total)

# Plot

df_total.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='green')

plt.title('Total Immigration From Norway, Denmark & Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
# plt.show()

# BUBBLE PLOTS ----------------------------------------------------------------------------------------------

# Immigration from Argentina & Brazil form 1980-2013

# transposed dataframe
df_can_t = df_can[years].transpose()

# cast the Years (the index) to type int
df_can_t.index = map(int, df_can_t.index)

# let's label the index. This will automatically be the column name when we reset the index
df_can_t.index.name = 'Year'

# reset index to bring the Year in as a column
df_can_t.reset_index(inplace=True)

# print(df_can_t)

# normalize Brazil data
norm_brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())
# print(norm_brazil)

# normalize Argentina data
norm_argentina = (df_can_t['Argentina'] - df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())

# Plot

# Brazil
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,  # transparency
                    color='green',
                    s=norm_brazil * 2000 + 10,  # sets size of values based on norm_brazil
                    xlim=(1975, 2015)
                    )

# Argentina
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Argentina',
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10, # sets size of values based on norm_argentina
                    ax=ax0
                    )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')

# Immigration from China & India to Canada from 1980-2013
 
# normalized Chinese data
norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())
# normalized Indian data
norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())

# China
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='China',
                    figsize=(14, 8),
                    alpha=0.5, # transparency
                    color='green',
                    s=norm_china * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                    )

# India
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='India',
                    alpha=0.5,
                    color="blue",
                    s=norm_india * 2000 + 10,
                    ax = ax0
                    )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China and India from 1980 - 2013')
ax0.legend(['China', 'India'], loc='upper left', fontsize='x-large')

# PRINTS ----------------------------------------------------------------------------------------------------

# print(df_can.head())
# print(df_can.shape)

# check for latest version of Matplotlib
# print('Matplotlib version: ', mpl.__version__) # >= 2.0.0

# -----------------------------------------------------------------------------------------------------------