#Import and setup matplotlib:
# IMPORTS ----------------------------------------------------------------------------------------------

import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
from PIL import Image # converting images into arrays
from pywaffle import Waffle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle Charts
import seaborn as sns
import wordcloud

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib and seaborn
# print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# print('Seaborn version: ', sns.__version__)
# print('WordCloud version: ', wordcloud.__version__)

# DATA ------------------------------------------------------------------------------------------------

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

#set Country as index
df_can.set_index('Country', inplace=True)

# WAFFLE CHARTS ------------------------------------------------------------------------------------------

# new dataframe for these three countries 
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]
df_ci = df_can.loc[['China', 'India'], :]

# compute the proportion of each category with respect to the total
total_values = df_dsn['Total'].sum()
category_proportions = df_dsn['Total'] / total_values

width = 40 # width of chart
height = 10 # height of chart

total_num_tiles = width * height # total number of tiles

# compute the number of tiles for each category
tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)

# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width), dtype = np.uint)

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# Function

def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_dsn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    # np.arange(-.5, (width), 1): This part generates an array of values starting from -0.5, incrementing by 1, and stopping before reaching the value (width). The resulting array represents the positions where the ticks will be placed on the x-axis.
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()

width = 40 # width of chart
height = 10 # height of chart

categories = df_dsn.index.values # categories
values = df_dsn['Total'] # correponding values of categories

# For Canada & India
valuess = df_ci['Total']
categoriess = df_ci.index.values # categories

colormap = plt.cm.coolwarm # color map class

create_waffle_chart(categoriess, valuess, height, width, colormap)

#Set up the Waffle chart figure

# fig = plt.figure(FigureClass = Waffle,
#                  rows = 20, columns = 30, #pass the number of rows and columns for the waffle 
#                  values = df_dsn['Total'], #pass the data to be used for display
#                  cmap_name = 'tab20', #color scheme
#                  legend = {'labels': [f"{k} ({v})" for k, v in zip(df_dsn.index.values,df_dsn.Total)],
#                             'loc': 'lower left', 'bbox_to_anchor':(0,-0.1),'ncol': 3}
#                  #notice the use of list comprehension for creating labels 
#                  #from index and total of the dataset
#                 )

# #Display the waffle chart
# plt.show()

# WORD CLOUD ---------------------------------------------------------------------------------------------



# PRINTS ----------------------------------------------------------------------------------------------

# print(df_can.head())
# print(df_can.shape)
# print(df_dsn)
# print(pd.DataFrame({"Category Proportion": category_proportions}))
# print(f'Total number of tiles is {total_num_tiles}.')
# print(pd.DataFrame({"Number of tiles": tiles_per_category}))

# ------------------------------------------------------------------------------------------------------

