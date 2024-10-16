
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

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/launch_dash.csv'
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

# ========================== ========================== #

options=[{'label': 'All Sites', 'value': 'ALL'},{'label': 'site1', 'value': 'site1'}, ...]

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
    className='row1',
    children=[
        html.Div(
            className='table2', 
            children=[
                dcc.Dropdown(id='id',
                options=[
                    {'label': 'All Sites', 'value': 'ALL'},
                    {'label': 'site1', 'value': 'site1'},
                ],
                value='ALL',
                placeholder="place holder here",
                searchable=True
                ),
                dcc.Graph(
                    className='data',
                    # figure=fig_head
                )
            ]
        ),
        html.Div(
            className='table2', 
            children=[
                dcc.RangeSlider(id='id',
                min=0, max=10000, step=1000,
                marks={0: '0',
                       100: '100'},
                value=[min, max]),
                dcc.Graph(
                    className='data',
                    # figure=fig_head
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
                    # srcDoc=site_map_html
                    # ,style={'border': 'none', 'width': '1800px', 'height': '800px'}
                )
            ]
        )
    ]
)
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

# updated_path = 'data/launch_dash.csv'
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