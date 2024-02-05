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
import folium
from folium import plugins
import requests
import webbrowser

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib and seaborn
# print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# print('Seaborn version: ', sns.__version__)
# print('WordCloud version: ', wordcloud.__version__)

# INTRODUCTION TO FOLIUM -------------------------------------------------------------------------------

# define the world map
world_map = folium.Map()

# define the world map centered around Canada with a low zoom level
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)

# World map Mexico
world_map_mex = folium.Map(location=[19.5, -98.9], zoom_start=7.3)

# create a Cartodb dark_matter map of the world centered around Canada
world_map_dark = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Cartodb dark_matter')

# create a Cartodb positron map of the world centered around Canada
world_map_positron = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Cartodb positron')

# MAPS WITH MARKERS ----------------------------------------------------------------------------------

df_incidents = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv')

# print(df_incidents.head())

# get the first 100 crimes in the df_incidents dataframe
limit = 100
df_incidents = df_incidents.iloc[0:limit, :]
# print(df_incidents.shape)

# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

# instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(sanfran_map)

# loop through the 100 crimes and add each to the incidents feature group
# zip iterates over pairs of latitude and longitude
# for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
#     folium.vector_layers.CircleMarker(
#         [lat, lng],
#         radius=5, # define how big you want the circle markers to be
#         color='yellow',
#         fill=True,
#         popup=label,
#         fill_color='blue',
#         fill_opacity=0.6
#     ).add_to(sanfran_map)

# CLUSTERS APPROACH:
    
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
  folium.Marker(
      location=[lat, lng],
      icon=None,
      popup=label,
  ).add_to(incidents)

# OTHER WAY:

# add pop-up text to each marker on the map
# latitudes = list(df_incidents.Y)
# longitudes = list(df_incidents.X)
# labels = list(df_incidents.Category)

# for lat, lng, label in zip(latitudes, longitudes, labels):
#     folium.Marker([lat, lng], popup=label).add_to(sanfran_map)    

# add incidents to map
sanfran_map.add_child(incidents)

# CHOROPLETH MAPS ------------------------------------------------------------------------------------

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

# GeoJSON data
world_geo = r'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json' # geojson file

# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2)

# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
folium.Choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
).add_to(world_map)

# -----------------------------------------------------------------------------------------------------

# Save the map as an HTML file
# world_map.save("world_map.html")
# sanfran_map.save("sanfran_map.html")
# world_map_mex.save('Mex_map.html')
# world_map.save("choropleth_map.html")
# webbrowser.open("choropleth_map.html")
# df_incidents.to_csv('Police_Department_Incidents_2016.csv')