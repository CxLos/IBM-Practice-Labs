
# ========================== Imports ==========================

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import r2_score
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

# ========================== Load Data ==========================

# df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/telecust1000.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Exploration ========================== #

# Preview Data
# print(df.head())

# summarize the data
# print(df.describe())

# Value counts
# print(df['custcat'].value_counts())
# print(df['region'].value_counts())

# Feature Set
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  #.astype(float)
# print(X[0:5]) # Display the first 5 rows

# Target Set
y = df['custcat'].values
print(y[0:5]) # Display the first 5 rows

# Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5]) # Display the first 5 rows

# ========================== Data Visualization ========================== #

# Histogram for 'Income'
hist_income = (
    px.histogram(df, x='income')
    .update_layout(
        title='Histogram for Income',
        title_x=0.5,
        bargap=0,  # Adjust this value to control the space between bars
        xaxis=dict(showgrid=True, layer='above traces'),
        yaxis=dict(showgrid=True, layer='above traces')
    )
    .update_traces(marker=dict(line=dict(color='black', width=2)))  # Outline color and width
)

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('K-Nearest Neighbors (KNN)', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%202%20-%20Regression/linear.py',
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
                    id='graph1',
                  #  Histogram for 'CYLINDERS'
                  figure=hist_income

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 2
html.Div(
    className='row2',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 3
html.Div(
    className='row2',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 4
html.Div(
    className='row2',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row2',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  
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

# ================================ Export Data =============================== #

# updated_path = 'data/telecust1000.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================================================================ #