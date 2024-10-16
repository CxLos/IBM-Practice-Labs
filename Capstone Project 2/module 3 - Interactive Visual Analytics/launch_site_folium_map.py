
# =============================== Imports ============================= #

from math import sin, cos, sqrt, atan2, radians
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import matplotlib.pyplot as plt
import csv, sqlite3
import itertools
import folium
import seaborn as sns
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html

# ============================= Load Data ============================= #

# df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/spacex_geo.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Exploratory Data ========================== #

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.columns)
# print(df.dtypes)

# ========================== 1. Mark All Launch Sites on the Map ========================== #

# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long','class']]
# print(df.head())

# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10, width='100%', height='100%')

# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)

# Initiate the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
for index, site in launch_sites_df.iterrows():
    # Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
    circle = folium.Circle([site['Lat'], site['Long']], radius=1000, color='#d35400', fill=True).add_child(folium.Popup(site['Launch Site']))
    marker = folium.map.Marker(
        [site['Lat'], site['Long']],
        # Create an icon as a text label
        icon=DivIcon(
            icon_size=(20,20),
            icon_anchor=(0,0),
            html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % site['Launch Site'],
            )
        )
    site_map.add_child(circle)
    site_map.add_child(marker)

map_path = 'spacex_map.html'
map_file = os.path.join(script_dir, map_path)
site_map.save(map_file)
site_map_html = open(map_file, 'r').read()

# =============== 2. Mark Successful / Failed Launches for each site ================== #

marker_cluster = MarkerCluster()

# Create a new column in `launch_sites` dataframe called `marker_color` to store the marker colors based on the `class` value
launch_sites_df['marker_color'] = launch_sites_df['class'].apply(lambda x: 'red' if x == 0 else 'green')

# Function to assign color to launch outcome
def assign_marker_color(launch_outcome):
    if launch_outcome == 1:
        return 'green'
    else:
        return 'red'
    
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)

# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed, 
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, record in spacex_df.iterrows():
    # TODO: Create and add a Marker cluster to the site map
    marker = folium.Marker([record['Lat'], record['Long']], icon=folium.Icon(color='white', icon_color=record['marker_color']))

    marker_cluster.add_child(marker)

# ========= 3. Calculate the distances between a launch site to its proximities ========= #

# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Mark down a point on the closest coastline using MousePosition and calculate the distance between the coastline point and the launch site.
# e.g.,: Lat: 28.56367  Lon: -80.57163
# distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)

lat1 = 28.56367
lon1 = -80.57163
lat2 = 29.559684888503615
lon2 = -95.0830971930759

distance_coastline = calculate_distance(lat1, lon1, lat2, lon2)

# Coordinates of the closest coastline point
coastline_lat = lat2
coastline_lon = lon2
launch_site_lat = lat1
launch_site_lon = lon1

# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property
# for example
# distance_marker = folium.Marker(
#    coordinate,
#    icon=DivIcon(
#        icon_size=(20,20),
#        icon_anchor=(0,0),
#        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance),
#        )
#    )

distance_marker = folium.Marker(
    [coastline_lat, coastline_lon],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
        )
      )
                   
site_map.add_child(distance_marker)

# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
# lines=folium.PolyLine(locations=coordinates, weight=1)
lines = folium.PolyLine(locations=[[launch_site_lat, launch_site_lon], [coastline_lat, coastline_lon]], weight=1)

site_map.add_child(lines)

# ========================== DataFrame Table ========================== #

fig_head = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(df.columns),
        fill_color='paleturquoise',
        align='left',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='lavender',
        align='left',
        height=25,  # Adjust the height of the cells
        # line=dict(color='black', width=1),  # Add border to cells
        font=dict(size=12)  # Adjust font size
    )
)])

fig_head.update_layout(
    margin=dict(l=50, r=50, t=30, b=40),  # Remove margins
    height=400,
    # width=1500,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('SpaceX Launch Records Dashboard', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/Capstone%20Project%202/module%202%20-%20EDA/eda_dataviz.py',
        className='btn')
    ]),

# Data Table
html.Div(
    className='row0',
    children=[
        html.Div(
            className='table',
            children=[
                html.H1(
                    className='table-title',
                    children='SpaceX Launch Records Data Table'
                )
            ]
        ),
        html.Div(
            className='table2', 
            children=[
                dcc.Graph(
                    className='data',
                    figure=fig_head
                )
            ]
        )
    ]
),

html.Div(
    className='row3',
    children=[
        html.Div(
            className='graph5',
            children=[
                html.H1(
                    'Spacex Houston', 
                    className='zip'
                ),
                html.Iframe(
                    className='folium',
                    id='folium-map',
                    srcDoc=site_map_html
                    ,style={'border': 'none', 'width': '1800px', 'height': '800px'}
                )
            ]
        )
    ]
)
])

if __name__ == '__main__':
    app.run_server(debug=
                   True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/spacex_geo.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #

# pip install dash --upgrade
# pip install dash-core-components --upgrade
# pip install dash-html-components --upgrade
# pip install dash-renderer --upgrade

# ========================================================================== #

# git rm --cached "12. Machine Learning with Python/module 3 - Classification/data/yellow_tripdata.csv"
# git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch "12. Machine Learning with Python/module 3 - Classification/data/yellow_tripdata.csv"' --prune-empty --tag-name-filter cat -- --all

# git push origin --force --all
# git push origin --force --tags