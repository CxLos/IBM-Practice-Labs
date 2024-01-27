# IMPORTS --------------------------------------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
from io import StringIO
# from js import fetch
import io

# FETCHING DATA ----------------------------------------------------------------------------------------------

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv"

# resp = await fetch(URL)
# text = io.BytesIO((await resp.arrayBuffer()).to_py())

response = requests.get(URL)
data = StringIO(response.text)
df_can = pd.read_csv(data)

# Set index to country
df_can.set_index('Country', inplace=True)

# Store years in a variable
years = list(map(str, range(1980, 2014)))
# years = np.arange(1980,2014)

# LINE PLOT --------------------------------------------------------------------------------------------------

#creating df with only years columns from 1980 - 2013
df_line = df_can[years]
# print(df_line)

#Applying sum to get total immigrants year-wise
total_immigrants = df_line.sum()
# print(total_immigrants)

#Create figure and axes
# fig, ax = plt.subplots()

# #Changing the index type to integer
# total_immigrants.index = total_immigrants.index.map(int)

# # Customizing the appearance of Plot
# ax.plot(total_immigrants, 
#         marker='s', #Including markers in squares shapes
#         markersize=5, #Setting the size of the marker
#         color='green', #Changing the color of the line
#         linestyle="dotted") #Changing the line style to a Dotted line

# #Setting up the Title
# ax.set_title('Immigrants between 1980 to 2013') 
# ax.set_xlabel('Years')
# ax.set_ylabel('Total Immigrants')
# ax.legend(['Immigrants'])

# #limits on x-axis
# plt.xlim(1975, 2015)  #or ax.set_xlim()

# #Enabling Grid
# plt.grid(True)  #or ax.grid()
# plt.show()

# Line Graph of Immigration from Haiti from 1980-2013

# df_can.reset_index(inplace=True)
haiti = df_can.loc['Haiti']
haiti = haiti[years].T
haiti.index = haiti.index.map(int)
# print(haiti)

#Plotting the line plot on the data
# fig, ax = plt.subplots()

# ax.plot(haiti)

# ax.set_title('Immigrants from Haiti between 1980 to 2013') 
# ax.set_xlabel('Years')
# ax.set_ylabel('Number of Immigrants')

# #Enabling Grid
# ax.grid()
# ax.legend(["Immigrants"]) #or ax.legend()
# ax.annotate('2010 Earthquake',xy=(2000, 6000))
# plt.show()

# SCATTER PLOT ----------------------------------------------------------------------------------------------

#Create figure and axes
# fig, ax = plt.subplots(figsize=(8, 4))

# total_immigrants.index = total_immigrants.index.map(int)

# # Customizing Scatter Plot 
# ax.scatter(total_immigrants.index, total_immigrants, 
#            marker='o', #setting up the markers
#            s = 20, #setting up the size of the markers
#            color='darkblue') #the color for the marker

# #add title 
# plt.title('Immigrants between 1980 to 2013') 
# #add labels 
# plt.xlabel('Years')
# plt.ylabel('Total Immigrants') 
# #including grid
# plt.grid(True)

# #Legend at upper center of the figure
# ax.legend(["Immigrants"], loc='upper center')

# plt.show()

# BAR PLOT --------------------------------------------------------------------------------------------------

#Sorting the dataframe on 'Total' in descending order
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

# get the top 5 entries with head function
df_top5 = df_can.head()

#resetting the index back to original way
df_bar_5=df_top5.reset_index()
# print(df_bar_5)

#Creating alist of names of the top 5 countries
label=list(df_bar_5.Country)
# label[2]='UK'
# print(label)

# fig, ax = plt.subplots(figsize=(10, 4))

# ax.bar(label,df_bar_5['Total'], label=label)
# ax.set_title('Immigration Trend of Top 5 Countries')
# ax.set_ylabel('Number of Immigrants')
# ax.set_xlabel('Years')

# plt.show()

# 5 Countries that contributed the least

df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

# get the top 5 entries with head function
df_top5 = df_can.tail()

#resetting the index back to original way
df_bar_5=df_top5.reset_index()
# print(df_bar_5)

#Creating alist of names of the top 5 countries
label=list(df_bar_5.Country)
# print(label)

# fig, ax = plt.subplots(figsize=(10, 4))

# ax.bar(label,df_bar_5['Total'], label=label)
# ax.set_title('Immigration Trend of Bottom 5 Countries')
# ax.set_ylabel('Number of Immigrants')
# ax.set_xlabel('Years')

# plt.show()

# HISTOGRAM ------------------------------------------------------------------------------------------------

df_country = df_can.groupby(['Country'])['2013'].sum().reset_index()

#Create figure and axes
# fig, ax = plt.subplots(figsize=(10, 4))

# # create histogram of the values in the 2013 column
# count = ax.hist(df_country['2013'])

# ax.hist(df_country['2013'])
# ax.set_title('New Immigrants in 2013') 
# ax.set_xlabel('Number of Immigrants')
# ax.set_ylabel('Number of Countries')
# ax.set_xticks(list(map(int,count[2])))
# ax.legend(['Immigrants'])

# plt.show()

# Histogram for Denmark, Norwary and Sweden

df = df_can.groupby(['Country'])[years].sum()
df_dns = df.loc[['Denmark', 'Norway', 'Sweden'], years]
# print(df_dns)
df_dns = df_dns.T

#Create figure and axes
# fig, ax = plt.subplots(figsize=(10, 4))

# ax.hist(df_dns)
# ax.set_title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013') 
# ax.set_xlabel('Number of Immigrants')
# ax.set_ylabel('Number of Years')
# ax.legend(['Denmark', 'Norway', 'Sweden'])

# plt.show()

# Immigration Distribution from China & India for 2000-2013

# year1 = list(map(str, range(2000,2014)))

# df_ci = df_can.loc[['China', 'India']]
# df_ci = df_ci[year1].T
# # print(df_ci)

# fig, ax = plt.subplots(figsize=(10, 4))

# ax.hist(df_ci)
# ax.set_title('Immigration from China and India 2000 - 2013') 
# ax.set_xlabel('Number of Immigrants')
# ax.set_ylabel('Number of Years')
# ax.legend(['China', 'India',])

# plt.show()

# PIE CHART ------------------------------------------------------------------------------------------------

#  Total Immigrants

# Immigration from 1980 - 1985
# fig,ax=plt.subplots()

#Pie on immigrants
# ax.pie(total_immigrants[0:5], labels=years[0:5], 
#        colors = ['gold','blue','lightgreen','coral','cyan'],
#        autopct='%1.1f%%',explode = [0,0,0,0,0.1]) #using explode to highlight the lowest 

# ax.set_aspect('equal')  # Ensure pie is drawn as a circle

# plt.title('Distribution of Immigrants from 1980 to 1985')
# #plt.legend(years[0:5]), include legend, if you donot want to pass the labels
# plt.show()

# Immigration by continent

df_con = df_can.groupby('Continent')['Total'].sum().reset_index()

# print(df_con)

label = list(df_con.Continent)
# print(label)
label[3] = 'LAC'
label[4] = 'NA'

# fig, ax = plt.subplots()

# ax.pie(df_con['Total'], 
#       #  labels=label,
#        autopct='%1.1f%%',
#        pctdistance=1.25)

# plt.title('Distribution of Immigrants from 1980 to 1985')
# plt.legend(label, loc='upper left'), # include legend, if you donot want to pass the labels
# plt.show()

# SUB-PLOTTING ---------------------------------------------------------------------------------------------

# Create a figure with two axes in a row
# fig, axs = plt.subplots(1, 2, sharey=True)

#Plotting in first axes - the left one
# axs[0].plot(total_immigrants)
# axs[0].set_title("Line plot on immigrants")

# #Plotting in second axes - the right one
# axs[1].scatter(total_immigrants.index, total_immigrants)
# axs[1].set_title("Scatter plot on immigrants")

# axs[0].set_ylabel("Number of Immigrants")
# fig.suptitle('Subplotting Example', fontsize=15)

# # Adjust spacing between subplots
# fig.tight_layout()

# plt.show()

# Create a figure with 2 axes - one row, two columns

# total_immigrants.index = total_immigrants.index.map(int)

# # df_can.reset_index(inplace=True)
# fig = plt.figure(figsize=(8,4))

# # Add the first subplot (top-left)
# axs1 = fig.add_subplot(1, 2, 1)

# #Plotting in first axes
# axs1.plot(total_immigrants)
# axs1.set_title("Line plot on immigrants")

# # Add the second subplot (top-right)
# axs2 = fig.add_subplot(1, 2, 2)

# #Plotting in second axes - the right one
# axs2.bar(total_immigrants.index, total_immigrants) 
# axs2.set_title("Bar plot on immigrants")
# fig.suptitle('Subplotting Example', fontsize=15)

# # Adjust spacing between subplots
# fig.tight_layout()

# # print(total_immigrants)
# plt.show()

# Create a figure with Four axes - two rows, two columns

# total_immigrants.index = total_immigrants.index.map(int)

# fig = plt.figure(figsize=(10, 10))

# # Add the first subplot (top-left)
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.plot(total_immigrants)
# ax1.set_title('Plot 1 - Line Plot')

# # Add the second subplot (top-right)
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.scatter(total_immigrants.index, total_immigrants)
# ax2.set_title('Plot 2 - Scatter plot')

# # Add the third subplot (bottom-left)
# ax3 = fig.add_subplot(2, 2, 3)
# ax3.hist(df_dns)
# ax3.set_title('Plot3 - Histogram') 
# ax3.set_xlabel('Number of Immigrants')
# ax3.set_ylabel('Number of Years')

# # Add the fourth subplot (bottom-right)
# ax4 = fig.add_subplot(2, 2, 4)
# ax4.pie(total_immigrants[0:5], labels=years[0:5], 
#         colors = ['gold','blue','lightgreen','coral','cyan'],
#         autopct='%1.1f%%')
# ax4.set_aspect('equal')  
# ax4.set_title('Plot 5 - Pie Chart')

# #Adding a Title for the Overall Figure
# fig.suptitle('Four Plots in a Figure Example', fontsize=15)

# # Adjust spacing between subplots
# fig.tight_layout()

# plt.show()

# PRINTS ----------------------------------------------------------------------------------------------------

# print(df_can.head())
# print('Years List:', years)
# print('data dimensions:', df_can.shape)

# check for latest version of Matplotlib
# print('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# -----------------------------------------------------------------------------------------------------------