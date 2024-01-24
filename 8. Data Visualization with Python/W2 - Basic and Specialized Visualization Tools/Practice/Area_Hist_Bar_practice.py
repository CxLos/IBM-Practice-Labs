# IMPORTS ----------------------------------------------------------------------------------------------------
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# optional: for ggplot-like style
mpl.style.use('ggplot')  
# check for latest version of Matplotlib
# print('Matplotlib version: ', mpl.__version__) # >= 2.0.0

# FETCHING DATA ----------------------------------------------------------------------------------------------

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

# Set Index
df_can.set_index('Country', inplace=True)

# Check dimensions
# print('data dimensions:', df_can.shape)

# Variable to store the years
years = list(map(str, range(1980, 2014)))

# Check for full list of available colors in matplotlib:
# for name, hex in mpl.colors.cnames.items():
#     print(name, hex)

# AREA PLOTS -------------------------------------------------------------------------------------------------

# Sort df in descending order
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

# get the top 5 entries
df_top5 = df_can.head()

# transpose the dataframe to invert the rows and columns
df_top5 = df_top5[years].transpose()
# print(df_top5)

# Change the index values of df_top5 to type integer for plotting
# alpha = transparency
# stacked = like do you want it faded and overlapping or hard colors one over the other.
# figsize = width_in, length_in 
df_top5.index = df_top5.index.map(int)
# df_top5.plot(kind='area',
#              alpha=0.25, 
#              stacked=False,
#              figsize=(20, 10))  # pass a tuple (x, y) size

# plt.title('Immigration Trend of Top 5 Countries')
# plt.ylabel('Number of Immigrants')
# plt.xlabel('Years')
# plt.show()

# Plotting using the Artist Layer (Obj oriented method). This is more flexible and transparent and better suited for advanced plots

# ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))

# ax.set_title('Immigration Trend of Top 5 Countries')
# ax.set_ylabel('Number of Immigrants')
# ax.set_xlabel('Years')

# Stacked Area Plot of the 5 countries that contributed the least to Canadian Immigration using the scripting layer.

df_bottom_5 = df_can.tail(5)
df_bottom_5 = df_bottom_5.dropna()
df_bottom_5 = df_bottom_5[years].transpose()
df_bottom_5.index = df_bottom_5.index.map(int)
# print(type(df_bottom_5.index))
# print(df_bottom_5.head())

# Plot
# df_bottom_5.plot(kind='area',
#              alpha=0.55, 
#              stacked=False,
#              figsize=(20, 10))  # pass a tuple (x, y) size

# plt.title('Immigration Trend of Bottom 5 Countries')
# plt.ylabel('Number of Immigrants')
# plt.xlabel('Years')
# plt.show()

# Stacked Area Plot of the 5 countries that contributed the least to Canadian Immigration using the artist layer.

# ax = df_bottom_5.plot(kind='area', alpha=0.55, figsize=(20, 10))

# ax.set_title('Immigration Trend of Bottom 5 Countries')
# ax.set_ylabel('Number of Immigrants')
# ax.set_xlabel('Years')
# plt.show()

# HISTOGRAMS ---------------------------------------------------------------------------------------------

# Quickly check stats for 2013
# print(df_can['2013'].head())

# np.histogram returns 2 values
count, bin_edges = np.histogram(df_can['2013'])

# frequency count (how many results in each bin)
# print(count) 
# Result:
# [178  11   1   2   0   0   0   0   1   2]

# bin ranges, default = 10 bins
# print(bin_edges) 
# Result:
# [    0.   3412.9  6825.8 10238.7 13651.6 17064.5 20477.4 23890.3 27303.2 30716.1 34129. ]

# Plot using scripting layer

# df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
# # add a title to the histogram
# plt.title('Histogram of Immigration from 195 Countries in 2013')
# # add y-label
# plt.ylabel('Number of Countries')
# # add x-label
# plt.xlabel('Number of Immigrants')
# plt.show()

# Immigration Distribution for Denmark, Norway, and Sweden for 1980 - 2013

df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
# print(df_t.head())

count, bin_edges = np.histogram(df_t, 15)
# print(count)
# print(bin_edges)

#  first bin value is 31.0, set minimum x-axis value to first bin edge[] - 0. 
xmin = bin_edges[0] - 0   
#  last bin value is 308.0, adding buffer of 10 for aesthetic purposes
xmax = bin_edges[-1] + 0

# un-stacked histogram
# df_t.plot(kind ='hist', 
#           figsize=(10, 6),
#           bins=15,
#           alpha=0.6,
#           xticks=bin_edges,
#           color=['coral', 'dodgerblue', 'mediumseagreen'],
#           stacked=True,
#           xlim=(xmin, xmax)
#          )

# plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
# plt.ylabel('Number of Years')
# plt.xlabel('Number of Immigrants')
# plt.show()

# print(df_can.loc[['Denmark', 'Norway', 'Sweden'], years])

# Display Immigration distribution for Greece, Albania, and Bulgaria from 1980-2013

df_b = df_can.loc[['Greece', 'Albania', 'Bulgaria'],years].transpose()
count, bin_edges = np.histogram(df_b, 15)

# df_b.plot(kind ='hist', 
#           figsize=(10, 6),
#           bins=15,
#           alpha=0.35,
#           xticks=bin_edges,
#           color=['coral', 'dodgerblue', 'mediumseagreen'],
#           stacked=True,
#          )

# plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013')
# plt.ylabel('Number of Years')
# plt.xlabel('Number of Immigrants')
# plt.show()


# BAR CHARTS --------------------------------------------------------------------------------------------------

# Vertical Bar Plot
df_iceland = df_can.loc['Iceland', years]
# print(df_iceland.head())

# rotate the xticks(labelled points on x-axis) by 90 degrees
# df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)

# plt.xlabel('Year') 
# plt.ylabel('Number of immigrants')
# plt.title('Icelandic immigrants to Canada from 1980 to 2013') 

# # Use plt.annotate() to add an arrow between 2 points on the plot.
# plt.annotate('',  # s: str. Will leave it blank for no text
#              xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
#              xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
#              xycoords='data',  # specified coordinates above are from the data we are using
#              arrowprops=dict(arrowstyle='wedge', # arrow style is pointed
#                              connectionstyle='arc3', # style of connection between arrow and annotated point
#                              color='blue', # color
#                              lw=2) # line width
#              )

# # Annotate Text
# plt.annotate('2008 - 2011 Financial Crisis',  # text to display
#              xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
#              rotation=72.5,  # based on trial and error to match the arrow
#              va='bottom',  # vertically align text to the bottom of annotation point
#              ha='left',  # horizontally align text to left of annotation point
#              )

# plt.show()

# Horizontal Bar Plot (barh)

# Immigration to Canada from top 15 countries from 1980-2013.

df_top15 = df_can.head(15)
# df_top15 = df_top15[years].transpose()
# df_top15.index = df_top15.index.map(int)
# print(df_top15)

df_top15.plot(kind='barh', figsize=(12,12))

plt.xlabel('Year') 
plt.ylabel('Number of immigrants')
plt.title('Immigration to Canada from top 15 countries') 
plt.show()


# PRINTS ------------------------------------------------------------------------------------------------------

# print(df_can.head())

# Get all available arrowstyles
all_arrowstyles = patches.ArrowStyle.get_styles()
# Print the list of arrowstyles
# print(all_arrowstyles)

# Get all available connection styles
all_connection_styles = patches.ConnectionStyle.get_styles()
# Print the list of connection styles
# print(all_connection_styles)

# Get all available plot styles
all_styles = plt.style.available
# Print the list of plot styles
# print(all_styles)

# df_can.to_excel('Canada_Immigration4.xlsx')