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
from wordcloud import WordCloud, STOPWORDS
import urllib.request

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

# create_waffle_chart(categoriess, valuess, height, width, colormap)

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

# Download the file for Alice in Wonderland
alice_novel = urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt').read().decode("utf-8")

# Create mask
alice_mask = np.array(Image.open(urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')))

# The line `stopwords = set(STOPWORDS)` is creating a set of stopwords. Stopwords are commonly used
# words (such as "the", "and", "is", etc.) that are often removed from text data because they do not
# carry significant meaning. In this case, the set of stopwords is being used for creating a word
# cloud from the text data.
stopwords = set(STOPWORDS)
stopwords.add('said') # add the words said to stopwords

# instantiate a word cloud object
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# fig = plt.figure(figsize=(14, 18))

# display the word cloud
# plt.imshow(alice_wc, cmap=plt.cm.gray, interpolation='bilinear')
# plt.imshow(alice_wc, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Generate sample text data from Canada df
total_immigration = df_can['Total'].sum()
total_immigration

max_words = 90
word_string = ''
for country in df_can.index.values:
     # check if country's name is a single-word name
    if country.count(" ") == 0:
        # How many words to generate per country
        repeat_num_times = int(df_can.loc[country, 'Total'] / total_immigration * max_words)
        # Concatenate country's name to word string equal to the amt above.
        word_string = word_string + ((country + ' ') * repeat_num_times)

# print(word_string)

# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

# display the cloud
# plt.figure(figsize=(14, 18))

# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# PLOTTING WITH SEABORN -------------------------------------------------------------------------------

df_can['Continent'].unique()
df_can1 = df_can.replace('Latin America and the Caribbean', 'L-America')
df_can1 = df_can1.replace('Northern America', 'N-America')
df_Can2=df_can1.groupby('Continent')['Total'].mean()
# print(df_Can2)

# Count Plot is like histogram but for categorical values
# plt.figure(figsize=(15, 10))
# sns.countplot(x='Continent', data=df_can1)
# sns.barplot(x='Continent', y='Total', data=df_can1)
# plt.show()

# REGRESSION PLOTS -----------------------------------------------------------------------------------

years = list(map(str, range(1980, 2014)))
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

# rename columns
df_tot.columns = ['year', 'total']

# print(df_tot.head())

# plt.figure(figsize=(15, 10))

# sns.set(font_scale=1.5)
# sns.set_style('whitegrid')

# ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', 
#                  scatter_kws={'s': 200,'edgecolors': 'black'}) # set size to 200
# ax.set(xlabel='Year', ylabel='Total Immigration')
# ax.set_title('Total Immigration to Canada from 1980 - 2013')
# plt.show()

# Immigration from Norway, Sweden, and Denmark
    
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

# create df_total by summing across three countries for each year
df_total = pd.DataFrame(df_countries.sum(axis=1))

# reset index in place
df_total.reset_index(inplace=True)

# rename columns
df_total.columns = ['year', 'total']

# change column year from string to int to create scatter plot
df_total['year'] = df_total['year'].astype(int)

# define figure size
plt.figure(figsize=(15, 10))

# define background style and font size
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

# generate plot and add title and axes labels
ax = sns.regplot(x='year', y='total', data=df_total, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigrationn from Denmark, Sweden, and Norway to Canada from 1980 - 2013')
# plt.show()

print(df_total)
# print(df_countries)

# PRINTS ----------------------------------------------------------------------------------------------

# print(df_can.head())
# print(df_can.shape)
# print(df_dsn)
# print(pd.DataFrame({"Category Proportion": category_proportions}))
# print(f'Total number of tiles is {total_num_tiles}.')
# print(pd.DataFrame({"Number of tiles": tiles_per_category}))

# ------------------------------------------------------------------------------------------------------

