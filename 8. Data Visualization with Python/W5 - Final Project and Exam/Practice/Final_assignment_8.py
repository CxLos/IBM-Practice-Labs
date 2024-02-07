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

df_line = df.groupby(df['Year'])['Automobile_Sales'].mean()

plt.figure(figsize=(10, 6))
df_line.plot(kind = 'line')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Automobile Sales Over the Years')
plt.xticks(list(range(1980,2024)), rotation = 75)
plt.text(1982, 650, '1981-82 Recession')
# plt.text(......, ..., '..............')
plt.legend()
plt.show()

# 1.2 Plot different lines for categories of vehicle type and analyse the trend to answer the question Is there a noticeable difference in sales trends between different vehicle types during recession periods?

df_Mline = df.groupby(['Year','Vehicle_Type'], as_index=False)['Automobile_Sales'].sum()
df_Mline.set_index('Year', inplace=True)
df_Mline = df_Mline.groupby(['Vehicle_Type'])['Automobile_Sales']
df_Mline.plot(kind='line')

plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Sales Trend Vehicle-wise during Recession')
plt.legend()
plt.show()

# 1.3 Use the functionality of **Seaborn Library** to create a visualization to compare the sales trend per vehicle type for a recession period with a non-recession period.

df1 = df.groupby('Recession')['Automobile_Sales'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession',  data=df1)
plt.xlabel('Recession Status')
plt.ylabel('Sales')
plt.title('Average Automobile Sales during Recession and Non-Recession')
plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
plt.show()

# 1.4 Use sub plotting to compare the variations in GDP during recession and non-recession period by developing line plots for each period.



# 1.5 Develop a Bubble plot for displaying the impact of seasonality on Automobile Sales.



# 1.6 Use the functionality of Matplotlib to develop a scatter plot to identify the correlation between average vehicle price relate to the sales volume during recessions. From the data, develop a scatter plot to identify if there a correlation between consumer confidence and automobile sales during recession period? 



# Plot another scatter plot and title it as 'Relationship between Average Vehicle Price and Sales during Recessions'



# 1.7 Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession and non-recession periods.



# 1.8 Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession and non-recession periods.



# 1.9 Develop a lineplot to analyse the effect of the unemployment rate on vehicle type and sales during the Recession Period.



# 1.10 Create a map on the hightest sales region/offices of the company during recession period


# -------------------------------------------- PART II -----------------------------------------------

# path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/us-states.json'

# df10 = pd.read_json(path)

# PRINTS ----------------------------------------------------------------------------------------------

# print(df.head())
# print(df.describe())
# print(df.columns)
# print(df1.head())
# print()
# print()
# print()

# SAVE ----------------------------------------------------------------------------------------------

# df.to_csv('historical_automobile_sales.csv')
# df.to_json('us-states.json')
# --------------------------------------------------------------------------------------------------