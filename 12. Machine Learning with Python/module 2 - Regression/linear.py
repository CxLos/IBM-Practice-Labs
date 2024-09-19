# ========================== Imports ==========================

import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

# ========================== Load Data ==========================

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/fuel_consumption_C02.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)
# print(df.head())

# print(current_dir)
# print(script_dir)

# ========================== Data Exploration ========================== #

# summarize the data
# print(df.describe())

# features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz_hist_cylinder = px.histogram(viz, x='CYLINDERS')
viz_hist_enginesize = px.histogram(viz, x='ENGINESIZE')
viz_hist_co2emissions = px.histogram(viz, x='CO2EMISSIONS')
viz_hist_fuelconsumption = px.histogram(viz, x='FUELCONSUMPTION_COMB')

# Enable grid lines
# for fig in [viz_hist_cylinder, viz_hist_enginesize, viz_hist_co2emissions, viz_hist_fuelconsumption]:
#     fig.update_layout(
#         xaxis=dict(showgrid=True),
#         yaxis=dict(showgrid=True)
#     )

# ===========================Dash App =========================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Simple Linear Regression', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs',
        className='btn')
    ]),

# ROW 1
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(
                  #  Histogram for 'CYLINDERS'
                  figure=viz_hist_cylinder
                    .update_layout(
                      title='Histogram for CYLINDERS',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                    # Outline color and width
                    .update_traces(marker=dict(line=dict(color='black', width=1)))  
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    # Histogram for 'ENGINESIZE'
                  figure=viz_hist_enginesize
                  .update_layout(
                      title='Histogram for CYLINDERS',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                     # Outline color and width
                    .update_traces(marker=dict(line=dict(color='black', width=1))) 
                )
            ]
        )
    ]
),

# ROW 2
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(
                  #  Histogram for 'CO2EMISSIONS'                  
                  figure=viz_hist_co2emissions
                  .update_layout(
                      title='Histogram for CYLINDERS',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                    #  Outline color and width
                    .update_traces(marker=dict(line=dict(color='black', width=1)))
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    #  Histogram for 'FUELCONSUMPTION_COMB'
                  figure=viz_hist_fuelconsumption
                  .update_layout(
                      title='Histogram for CYLINDERS',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                    # Outline color and width
                    .update_traces(marker=dict(line=dict(color='black', width=1)))  
                )
            ]
        )
    ]
),
])

if __name__ == '__main__':
    app.run_server(debug=
                   True)
                #    False)

# -------------------------------------- Export Database -------------------------------------- #

# updated_path = 'data/fuel_consumption_C02.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ---------------------------------------------------------------------------------------------