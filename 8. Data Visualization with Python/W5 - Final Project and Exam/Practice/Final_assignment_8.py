# IMPORTS --------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
import datetime as dt

# DATA ------------------------------------------------------------------------------------------------

data = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

df = pd.read_csv(data)

# --------------------------------------------- PART I ----------------------------------------------

# 1.1 Develop a *Line chart* using the functionality of pandas to show how automobile sales fluctuate from year to year

# df_line = df.groupby(df['Year'])['Automobile_Sales'].mean()

# plt.figure(figsize=(10, 6))
# df_line.plot(kind = 'line')
# plt.xlabel('Year')
# plt.ylabel('Sales')
# plt.title('Automobile Sales Over the Years')
# plt.xticks(list(range(1980,2024)), rotation = 75)
# plt.text(1982, 650, '1981-82 Recession')
# # plt.text(......, ..., '..............')
# plt.legend()
# plt.show()

# 1.2 Plot different lines for categories of vehicle type and analyse the trend to answer the question Is there a noticeable difference in sales trends between different vehicle types during recession periods?

# df_Mline = df.groupby(['Year','Vehicle_Type'], as_index=False)['Automobile_Sales'].sum()
# df_Mline.set_index('Year', inplace=True)
# df_Mline = df_Mline.groupby(['Vehicle_Type'])['Automobile_Sales']
# df_Mline.plot(kind='line')

# plt.xlabel('Year')
# plt.ylabel('Sales')
# plt.title('Sales Trend Vehicle-wise during Recession')
# plt.legend()
# plt.show()

# 1.3 Use the functionality of **Seaborn Library** to create a visualization to compare the sales trend per vehicle type for a recession period with a non-recession period.

# df1 = df.groupby('Recession')['Automobile_Sales'].mean().reset_index()

# plt.figure(figsize=(10,6))
# sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession',  data=df1)
# plt.xlabel('Recession Status')
# plt.ylabel('Sales')
# plt.title('Average Automobile Sales during Recession and Non-Recession')
# plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
# plt.show()

# Group by vehicle type

# plt.figure(figsize=(10, 6))
# sns.barplot(x='Recession', y='Automobile_Sales', hue='Vehicle_Type', data=dd)
# plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
# plt.xlabel('Recession Status')
# plt.ylabel('Sales')
# plt.title('Vehicle-Wise Sales during Recession and Non-Recession Period')

# plt.show()

# 1.4 Use sub plotting to compare the variations in GDP during recession and non-recession period by developing line plots for each period.

# rec_data = df[df['Recession'] == 1]
# non_rec_data = df[df['Recession'] == 0]

# #Figure
# fig=plt.figure(figsize=(12, 6))

# # Axes
# ax0 = fig.add_subplot(1, 2, 1) 
# ax1 = fig.add_subplot(1,2,2 )

# sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession', ax=ax0)
# ax0.set_xlabel('Year')
# ax0.set_ylabel('GDP')
# ax0.set_title('GDP Variation during Recession Period')

# sns.lineplot(x='Year', y='GDP', data=non_rec_data, label='Recession', ax=ax1)
# ax1.set_xlabel('Year')
# ax1.set_ylabel('GDP')
# ax1.set_title('GDP Variation during Non-Recession Period')

# plt.tight_layout()
# plt.show()

# 1.5 Develop a Bubble plot for displaying the impact of seasonality on Automobile Sales.

# non_rec_data = df[df['Recession'] == 0]

# size=non_rec_data['Seasonality_Weight'] 

# sns.scatterplot(data=non_rec_data, x='Month', y='Automobile_Sales', size=size, hue='Seasonality_Weight')

# plt.xlabel('Month')
# plt.ylabel('Automobile_Sales')
# plt.title('Seasonality impact on Automobile Sales')

# plt.show()

# 1.6 Use the functionality of Matplotlib to develop a scatter plot to identify the correlation between average vehicle price relate to the sales volume during recessions. From the data, develop a scatter plot to identify if there a correlation between consumer confidence and automobile sales during recession period? 

# rec_data = df[df['Recession'] == 1]
# non_rec_data = df[df['Recession'] == 0]

# plt.figure(figsize=(10, 6))

# plt.scatter(rec_data['Consumer_Confidence'], rec_data['Automobile_Sales'], label='Recession')
# plt.scatter(non_rec_data['Consumer_Confidence'], non_rec_data['Automobile_Sales'], label='Non Recession')

# plt.xlabel('Consumer Confidence')
# plt.ylabel('Automobile Sales')
# plt.title('Consumer Confidence and Automobile Sales during Recessions')
# plt.legend()
# plt.show()

# # Plot another scatter plot and title it as 'Relationship between Average Vehicle Price and Sales during Recessions'

# plt.figure(figsize=(10, 6))
# plt.scatter(non_rec_data['Price'], non_rec_data['Automobile_Sales'], color='blue', label='Non-Recession')
# plt.scatter(rec_data['Price'], rec_data['Automobile_Sales'], color='red', label='Recession')


# plt.xlabel('Average Vehicle Price')
# plt.ylabel('Automobile_Sales')
# plt.title('Relationship between Average Vehicle Price and Sales during Recessions')
# plt.legend()
# plt.show()

# 1.7 Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession and non-recession periods.

# rec_data = df[df['Recession'] == 1]
# non_rec_data = df[df['Recession'] == 0]

# RAtotal = rec_data['Advertising_Expenditure'].sum()
# NRAtotal = non_rec_data['Advertising_Expenditure'].sum()

# # Create a pie chart for the advertising expenditure 
# plt.figure(figsize=(8, 6))

# labels = ['Recession', 'Non-Recession']
# sizes = [RAtotal, NRAtotal]
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# plt.title('Advertising Expenditure of XYZAutomotive')

# plt.show()

# 1.8 Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession for each vehicle type.

# r_df = df[df['Recession'] == 1]

# # New Data
# exec_data = r_df[r_df['Vehicle_Type'] == 'Executivecar']
# med_data = r_df[r_df['Vehicle_Type'] == 'Mediumfamilycar']
# sml_data = r_df[r_df['Vehicle_Type'] == 'Smallfamilycar']
# sport_data = r_df[r_df['Vehicle_Type'] == 'Sports']
# mini_data = r_df[r_df['Vehicle_Type'] == 'Supperminicar']

# # Sizes
# exec_total = exec_data['Advertising_Expenditure'].sum()
# med_total = med_data['Advertising_Expenditure'].sum()
# sml_total = sml_data['Advertising_Expenditure'].sum()
# sport_total = sport_data['Advertising_Expenditure'].sum()
# mini_total = mini_data['Advertising_Expenditure'].sum()

# # pie Chart
# plt.figure(figsize=(8, 6))

# labels = ['Executivecar', 'Mediumfamilycar','Smallfamilycar','Sports','Supperminicar']
# sizes = [exec_total, med_total, sml_total, sport_total, mini_total]
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# plt.title('Advertising Expenditure of XYZAutomotive by Vehicle Type during')

# plt.show()

# 1.9 Develop a lineplot to analyse the effect of the unemployment rate on vehicle type and sales during the Recession Period.

# r_df = df[df['Recession'] == 1]

# sns.lineplot(data=r_df, x='unemployment_rate', y='Automobile_Sales',
#              hue='Vehicle_Type', style='Vehicle_Type', markers='o', err_style=None)
# plt.title('Automobile Sales During Recession')
# plt.ylim(0,850)
# plt.legend(loc=(0.05,.3))

# 1.10 Create a map on the hightest sales region/offices of the company during recession period


# -------------------------------------------- PART II -----------------------------------------------

# geo = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/us-states.json'

# r_df = df[df['Recession'] == 1]
# sales_by_city = r_df.groupby('City')['Automobile_Sales'].sum().reset_index()
# print(sales_by_city)

# world_map = folium.Map(location=[0, 0], zoom_start=2)

# # generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
# folium.Choropleth(
#     geo_data=geo,
#     data=df,
#     columns=['City', 'Automobile_Sales'],
#     key_on='feature.properties.name',
#     fill_color='YlOrRd', 
#     fill_opacity=0.7, 
#     line_opacity=0.2,
#     legend_name='Sales by Region During Recession',
#     reset=True
# ).add_to(world_map)

# PRINTS ----------------------------------------------------------------------------------------------

# print(df[df['Recession'] == 1])
# print(df.shape)
# print(df.describe())
# print(df.columns)
# print(df1.head())
# print(df10.head())
# print(df['City'].unique())
# print()

# SAVE ----------------------------------------------------------------------------------------------

# df.to_csv('historical_automobile_sales.csv')
# df.to_json('us-states.json')
# --------------------------------------------------------------------------------------------------