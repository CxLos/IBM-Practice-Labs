
# =============================== Imports ============================= #

from sklearn.cluster import KMeans 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D 
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import itertools
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# ============================= Load Data ============================= #

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/australia_weather_data.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Pre Processing ========================== #

# One-Hot Encoding
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

# ========================== K - Means ========================== #



# ========================== Train / Test Split ========================== #

# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

# ========================== Modeling ========================== #



# ========================== Data Visualization ========================== #



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
    width=2800,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Rain Prediction in Australia', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%206%20-%20Rain%20Prediciton%20in%20Australia/australia_rain_data.py',
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
                    children='Australia Weather Data Table'
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

# ROW 1
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(
                  # figure=
                # figure={'data': [scatter]}
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    # figure=
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
])

if __name__ == '__main__':
    app.run_server(debug=
                   True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/australia_weather_data.csv'
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