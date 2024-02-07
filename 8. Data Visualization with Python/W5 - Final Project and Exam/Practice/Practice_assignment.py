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

# DATA -----------------------------------------------------------------------------------------------

# data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Historical_Wildfires.csv")

data = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\8. Data Visualization with Python\W5 - Final Project and Exam\Data\historical_wildfires.csv'

df = pd.read_csv(data)

# Format date
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month

# PART 1 ---------------------------------------------------------------------------------------------

# 1. Change in Average estimated fire area over time

# plt.figure(figsize=(12, 6))

# df_new=df.groupby('Year')['Estimated_fire_area'].mean()

# # Grouped by year & month
# df_new=df.groupby(['Year','Month'])['Estimated_fire_area'].mean()

# df_new.plot(x=df_new.index, y=df_new.values)
# plt.xlabel('Year')
# plt.ylabel('Average Estimated Fire Area (km²)')
# plt.title('Estimated Fire Area over Time')
# plt.show()

# 3. Bar Plot by region

# plt.figure(figsize=(10, 6))
# sns.barplot(data=df, x='Region', y='Mean_estimated_fire_brightness')
# plt.xlabel('Region')
# plt.ylabel('Mean Estimated Fire Brightness (Kelvin)')
# plt.title('Distribution of Mean Estimated Fire Brightness across Regions')
# plt.show()

# 4. Pie chart

# plt.figure(figsize=(10, 6))
# region_counts = df.groupby('Region')['Count'].sum()
# plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%')
# plt.title('Percentage of Pixels for Presumed Vegetation Fires by Region')
# plt.axis('equal')
# plt.show()

# 6. Histogram 

# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Mean_estimated_fire_brightness', hue='Region', multiple='stack')
# plt.xlabel('Mean Estimated Fire Brightness (Kelvin)')
# plt.ylabel('Count')
# plt.title('Histogram of Mean Estimated Fire Brightness')
# plt.show()

# 8. Scatter PLot to find correlation between mean est fire radiative power and confidence level

# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df, x='Mean_confidence', y='Mean_estimated_fire_radiative_power')
# plt.xlabel('Mean Estimated Fire Radiative Power (MW)')
# plt.ylabel('Mean Confidence')
# plt.title('Mean Estimated Fire Radiative Power vs. Mean Confidence')
# plt.show()

# 9. 

# region_data = {'region':
#                ['NSW','QL','SA','TA','VI','WA','NT'], 
#                'Lat':[-31.8759835,-22.1646782,-30.5343665,-42.035067,-36.5986096,-25.2303005,-19.491411], 
#                'Lon':[147.2869493,144.5844903,135.6301212,146.6366887,144.6780052,121.0187246,132.550964]}

# reg=pd.DataFrame(region_data)

# # instantiate a feature group 
# aus_reg = folium.map.FeatureGroup()

# # Create a Folium map centered on Australia
# Aus_map = folium.Map(location=[-25, 135], zoom_start=4)

# # loop through the region and add to feature group
# for lat, lng, lab in zip(reg.Lat, reg.Lon, reg.region):
#     aus_reg.add_child(
#         folium.features.CircleMarker(
#             [lat, lng],
#             popup=lab,
#             radius=5, # define how big you want the circle markers to be
#             color='red',
#             fill=True,
#             fill_color='blue',
#             fill_opacity=0.6
#         )
#     )

# # add incidents to map
# Aus_map.add_child(aus_reg)

# PART II --------------------------------------------------------------------------------------------

#Create app
app = dash.Dash(__name__)

# Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True

# data
df =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Historical_Wildfires.csv')

#Extract year and month from the date column
df['Month'] = pd.to_datetime(df['Date']).dt.month_name() #used for the names of the months
df['Year'] = pd.to_datetime(df['Date']).dt.year

#Task 2.1 Add the Title to the Dashboard
app.layout = html.Div(children=
                      [html.H1('Australia Wildfire Dashboard',
                               style={'textAlign': 'center', 'color': '#503D36',
                                'font-size': 26}),
                                
# TASK 2: Add the radio items and a dropdown right below the first inner division
     #outer division starts
            html.Div([
                   # First inner divsion for  adding dropdown helper text for Selected Drive wheels
                    html.Div([
                            html.H2('Select Region:', style={'margin-right': '2em'}),
                    #Radio items to select the region
                    #dcc.RadioItems(['NSW','QL','SA','TA','VI','WA'], 'NSW', id='region',inline=True)]),
                    dcc.RadioItems([{"label":"New South Wales","value": "NSW"},
                                    {"label":"Northern Territory","value": "NT"},
                                    {"label":"Queensland","value": "QL"},
                                    {"label":"South Australia","value": "SA"},
                                    {"label":"Tasmania","value": "TA"},
                                    {"label":"Victoria","value": "VI"},
                                    {"label":"Western Australia","value": "WA"}],"NSW", id='region',inline=True)]),
                    #Dropdown to select year
                    html.Div([
                            html.H2('Select Year:', style={'margin-right': '2em'}),
                        dcc.Dropdown(df.Year.unique(), value = 2005,id='year')
                    ]),
#TASK 3: Add two empty divisions for output inside the next inner division. 
         #Second Inner division for adding 2 inner divisions for 2 output graphs
                    html.Div([
                
                        html.Div([ ], id='plot1'),
                        html.Div([ ], id='plot2')
                    ], style={'display': 'flex'}),
    ])

])

#TASK 4: Add the Ouput and input components inside the app.callback decorator.
#Place to add @app.callback Decorator
@app.callback([Output(component_id='plot1', component_property='children'),
               Output(component_id='plot2', component_property='children')],
               [Input(component_id='region', component_property='value'),
                Input(component_id='year', component_property='value')])

#TASK 5: Add the callback function.   
#Place to define the callback function .
def reg_year_display(input_region,input_year):  
   
    #first filter dataframe by the entered region
   region_data = df[df['Region'] == input_region]
  #  then by the entered year
   y_r_data = region_data[region_data['Year']==input_year]

    #Plot one - Monthly Average Estimated Fire Area   
   est_data = y_r_data.groupby('Month')['Estimated_fire_area'].mean().reset_index()
   fig1 = px.pie(est_data, values='Estimated_fire_area', names='Month', title="{} : Monthly Average Estimated Fire Area in year {}".format(input_region,input_year))   

     #Plot two - Monthly Average Count of Pixels for Presumed Vegetation Fires
   veg_data = y_r_data.groupby('Month')['Count'].mean().reset_index()
   fig2 = px.bar(veg_data, x='Month', y='Count', title='{} : Average Count of Pixels for Presumed Vegetation Fires in year {}'.format(input_region,input_year))    

  #  populate to html
   return [dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2) ]

# Run app
if __name__ == '__main__':
    app.run_server()

# PRINTS ----------------------------------------------------------------------------------------------

# print(df.head())
# print(df.columns)
# print(df.shape)
# print(df['Region'].unique())
# data.to_csv('historical_wildfires.csv')