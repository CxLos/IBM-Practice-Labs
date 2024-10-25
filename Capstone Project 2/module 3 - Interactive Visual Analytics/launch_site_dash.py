
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
from dash import dcc, html, Input, Output

# ============================= Load Data ============================= #

# df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")

# df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")

# df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = 'data/launch_dash.csv'
# data_path = 'data/spacex_data_pt1.csv'
data_path = 'data/spacex_data_pt2.csv'
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

# ========================== Columns ========================== #

# FlightNumber       0
# Date               0
# BoosterVersion     0
# PayloadMass        0
# Orbit              0
# LaunchSite         0
# Outcome            0
# Flights            0
# GridFins           0
# Reused             0
# Legs               0
# LandingPad        26
# Block              0
# ReusedCount        0
# Serial             0
# Longitude          0
# Latitude           0
# Class              0
# Year               0

# ========================== ========================== #

options=[{'label': 'All Sites', 'value': 'ALL'},{'label': 'site1', 'value': 'site1'}, ...]

# ========================== Graphs ========================== #

# Convert class column to str
# df['Class'] = df['Class'].astype(str)

# Calculate Success Ratio of each launch site where 0 = success and 1 = failure:
# Round to 2 decimal places
df_success = df.groupby(['LaunchSite', 'Class']).size().unstack() # we use size() instead of count() to include NaN values and unstack() to pivot the table
df_success['success_rate'] = round(df_success[0] / (df_success[0] + df_success[1]), 2)
df_success = df_success.reset_index()

# print(df_success.head())
# value counts for 'Class' column
# print(df['Class'].value_counts())

# 1. Scatter Plot of Flight Number vs. Launch Site
flight_launch_scatter = px.scatter(
    df, 
    x='FlightNumber', 
    y='LaunchSite', 
    color='Class', 

)

flight_launch_scatter.update_layout(
    title='Flight Number vs. Launch Site',
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

flight_launch_scatter.update_traces(
    hovertemplate='<b>Flight Number: %{x}</b><br>Launch Site: %{y}<br>Class: %{marker.color}<extra></extra>'
)

# 2. Scatter Plot of Payload vs. Launch Site
payload_launch_scatter = px.scatter(
    df,
    x='PayloadMass',
    y='LaunchSite',
    color='Class',
)

payload_launch_scatter.update_layout(
    title='Payload vs. Launch Site',
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

payload_launch_scatter.update_traces(
    hovertemplate='<b>Payload Mass:</b> %{x} kg<br><b>Launch Site:</b> %{y}<br><b>Class:</b> %{marker.color}<extra></extra>'
)

# 3. Scatter Plot Success Rate vs. Orbit Type
success_orbit_scatter = px.scatter(
    df,
    x='Orbit',
    y='Outcome',
    color='Class',
)

success_orbit_scatter.update_layout(
    title='Success Rate vs. Orbit Type',
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

success_orbit_scatter.update_traces(
    hovertemplate='<b>Orbit Type:</b> %{x}<br><b>Outcome:</b> %{y}<br><b>Class:</b> %{marker.color}<extra></extra>'
)

# 4. Scatter Plot Flight Number vs. Orbit Type
flight_orbit_scatter = px.scatter(
    df,
    x='FlightNumber',
    y='Orbit',
    color='Class',
)

flight_orbit_scatter.update_layout(
    title='Flight Number vs. Orbit Type',
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

flight_orbit_scatter.update_traces(
    hovertemplate='<b>Flight Number:</b> %{x}<br><b>Orbit:</b> %{y}<br><b>Class:</b> %{marker.color}<extra></extra>'
)

# 5. Scatter Plot Payload vs. Orbit Type
payload_orbit_scatter = px.scatter(
    df,
    x='PayloadMass',
    y='Orbit',
    color='Class',
)

payload_orbit_scatter.update_layout(
    title='Payload vs. Orbit Type',
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

payload_orbit_scatter.update_traces(
    hovertemplate='<b>Payload Mass:</b> %{x} kg<br><b>Orbit:</b> %{y}<br><b>Class:</b> %{marker.color}<extra></extra>'
)

# 6. Line Chart of Yearly Success Rate
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Year"] = df["Date"].dt.year
suc_rate = df.groupby("Year")["Class"].mean().reset_index()

yearly_success_rate = px.line(
    suc_rate,
    x='Year',
    y='Class',
)

yearly_success_rate.update_layout(
    title='Yearly Success Rate',
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

yearly_success_rate.update_traces(
    hovertemplate='<b>Year:</b> %{x}<br><b>Success Rate:</b> %{y:.2f}<extra></extra>'
)

# 7. Launch Success Count for all Sites Pie Chart

site_success = px.pie(
    df,
    names='LaunchSite',
    title='Launch Success Count for all Sites',
    hole=0.3
)

site_success.update_layout(
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

site_success.update_traces(
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
    textinfo='value',
    textfont_size=20
)

# piechart for success ratio of each launch site
success_rate = px.pie(
    df_success,
    names='LaunchSite',
    values='success_rate',
    title='Success Ratio of Each Launch Site',
    hole=0.3
)

success_rate.update_layout(
    title_x=0.5,
    font=dict(
    family='Calibri',
    size=17,
    color='black'
    )
)

success_rate.update_traces(
    hovertemplate='<b>%{label}</b><br>Success Ratio: %{value: .2%}<extra></extra>',
    textinfo='percent',
    texttemplate='%{percent:.2%}',
    textfont_size=20
)

# Scatter Plot of Payload vs. Launch Outcome
# Include Range Slider for Payload Mass
payload_outcome_scatter = px.scatter(
    df,
    x='PayloadMass',
    y='Outcome',
    color='Class',
)

payload_outcome_scatter.update_layout(
    title='Payload vs. Launch Outcome',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        title='Payload Range Slider',
        rangeslider=dict(
            visible=True
        ),
        type='linear'
    )
)

payload_outcome_scatter.update_traces(
    hovertemplate='<b>Payload Mass:</b> %{x} kg<br><b>Outcome:</b> %{y}<br><b>Class:</b> %{marker.color}<extra></extra>'
)


# ========================== Questions ========================== #

# 1. All Launch Sites
# print(df['LaunchSite'].value_counts())

# CCAFS SLC 40    55
# KSC LC 39A      22
# VAFB SLC 4E     13

# 2. Launch Site Names that begin with 'CCA'
# print(df[df['LaunchSite'].str.contains('CCA')]['LaunchSite'].unique())

# ['CCAFS SLC 40']

# 3. Display the total payload mass carried by boosters launched by NASA (CRS)
# query = pd.read_sql_query("""
#   SELECT SUM(PAYLOAD_MASS__KG_) 
#   FROM SPACEXTBL 
#   WHERE Customer='NASA (CRS)'
#   """, con)

#    SUM(PAYLOAD_MASS__KG_)
# 0                   45596

# 4. Display average payload mass carried by booster version F9 v1.1
# query = pd.read_sql_query("""
#   SELECT AVG(PAYLOAD_MASS__KG_)
#   FROM SPACEXTBL
#   WHERE Booster_Version='F9 v1.1'
#   """, con)

# print(query)

#    AVG(PAYLOAD_MASS__KG_)
# 0                  2928.4

# 5. List the date when the first succesful landing outcome in ground pad was acheived.
# query = pd.read_sql_query("""
#   SELECT MIN(Date)
#   FROM SPACEXTBL
#   WHERE Landing_Outcome='Success (ground pad)'
#   """, con)

# print(query)

#     MIN(Date)
# 0  2015-12-22

# 6. List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000

# query = pd.read_sql_query("""
#   SELECT Booster_Version, Landing_Outcome
#   FROM SPACEXTBL
#   WHERE Landing_Outcome='Success (drone ship)'
#   AND PAYLOAD_MASS__KG_ BETWEEN 4000 AND 6000
#   """, con)

# print(query)

#   Booster_Version       Landing_Outcome
# 0     F9 FT B1022  Success (drone ship)
# 1     F9 FT B1026  Success (drone ship)
# 2  F9 FT  B1021.2  Success (drone ship)
# 3  F9 FT  B1031.2  Success (drone ship)

# 7. List the total number of successful and failure mission outcomes
# query = pd.read_sql_query("""
#   SELECT Count(Mission_Outcome)
#   FROM SPACEXTBL
#   WHERE Mission_Outcome='Success' OR Mission_Outcome='Failure'
#   """, con)

# print(query)

#    Count(Mission_Outcome)
# 0                      98

# 8. List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
# query = pd.read_sql_query("""
#   SELECT Booster_Version, MAX(PAYLOAD_MASS__KG_)
#   FROM SPACEXTBL
#   WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL)
#   """, con)

# print(query)

#   Booster_Version  MAX(PAYLOAD_MASS__KG_)
# 0   F9 B5 B1048.4                   15600

# 9. List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.
# query = pd.read_sql_query("""
#   SELECT strftime('%B', (Date)) as Month, Landing_Outcome, Booster_Version, Launch_Site
#   FROM SPACEXTBL
#   WHERE Landing_Outcome='Failure (drone ship)'
#   AND strftime('%Y', (Date))='2015'
#   """, con)

# print(query)

#   Month       Landing_Outcome Booster_Version  Launch_Site
# 0  None  Failure (drone ship)   F9 v1.1 B1012  CCAFS LC-40
# 1  None  Failure (drone ship)   F9 v1.1 B1015  CCAFS LC-40

# 10. Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.
# query = pd.read_sql_query("""
#   SELECT Landing_Outcome, COUNT(Landing_Outcome) as Count
#   FROM SPACEXTBL
#   WHERE Date BETWEEN '2010-06-04' AND '2017-03-20'
#   GROUP BY Landing_Outcome 
#   ORDER BY Count DESC
#   """, con)

# print(query)

#           Landing_Outcome  Count
# 0              No attempt     10
# 1    Success (drone ship)      5
# 2    Failure (drone ship)      5
# 3    Success (ground pad)      3
# 4      Controlled (ocean)      3
# 5    Uncontrolled (ocean)      2
# 6     Failure (parachute)      2
# 7  Precluded (drone ship)      1

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

# Row 1
# html.Div(
#     className='row1',
#     children=[
#         html.Div(
#             className='table2', 
#             children=[
#                 dcc.Dropdown(id='id',
#                 options=[
#                     {'label': 'All Sites', 'value': 'ALL'},
#                     {'label': 'site1', 'value': 'site1'},
#                 ],
#                 value='ALL',
#                 placeholder="place holder here",
#                 searchable=True
#                 ),
#                 dcc.Graph(
#                     className='data',
#                     # figure=fig_head
#                 )
#             ]
#         ),
#         html.Div(
#             className='table2', 
#             children=[
#                 dcc.RangeSlider(id='id',
#                 min=0, max=10000, step=1000,
#                 marks={0: '0',
#                        100: '100'},
#                 value=[min, max]),
#                 dcc.Graph(
#                     className='data',
#                     # figure=fig_head
#                 )
#             ]
#         )
#     ]
# ),

# ROW 3
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    figure=flight_launch_scatter
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    figure=payload_launch_scatter
                )
            ]
        )
    ]
),

# ROW 4
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    figure=flight_orbit_scatter
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    figure=payload_orbit_scatter
                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    figure=yearly_success_rate
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    figure=success_orbit_scatter
                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    figure=site_success
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    figure=payload_outcome_scatter
                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    figure=success_rate
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    # figure=
                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    # figure=
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    # figure=
                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph3',
            children=[
                dcc.Graph(
                    # figure=
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                dcc.Graph(
                    # figure=
                )
            ]
        )
    ]
),

# Row 
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
                    # srcDoc=site_map_html
                    # ,style={'border': 'none', 'width': '1800px', 'height': '800px'}
                )
            ]
        )
    ]
),
])

# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))

def get_pie_chart(entered_site):
    filtered_df = df
    if entered_site == 'ALL':
        fig = px.pie(df, values='class', 
        names='pie chart names', 
        title='title')
        return fig
    else:
        return
        # return the outcomes piechart for a selected site

if __name__ == '__main__':
    app.run_server(debug=
                   True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/spacex_data_pt2.csv'
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