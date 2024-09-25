
# ================================ Imports ======================== #

import pandas as pd
import pylab as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error
import itertools
import base64
import os
import warnings
import gc, sys
from io import BytesIO
import dash
from dash import dcc, html
from dash.development.base_component import Component

# ================================ Data =========================== #

# raw_data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/yellow_tripdata_2019-06.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/yellow_tripdata.csv'
file_path = os.path.join(script_dir, data_path)
raw_data = pd.read_csv(file_path)

# ========================== Data Exploration ========================== #

# Preview Data
# print(churn_df.head())

# Summarize the data
# print("DF Shape:", churn_df.shape)
# print("DTypes: \n", churn_df.dtypes)
# print("Description: \n", churn_df.describe())
# print("Info:", churn_df.info())
# print("Columns:", churn_df.columns)

# ========================== Pre Processing ========================== #

#Reducing the data size to 100000 records
raw_data=raw_data.head(100000)

# some trips report 0 tip. it is assumed that these tips were paid in cash.
# for this study we drop all these rows
raw_data = raw_data[raw_data['tip_amount'] > 0]

# we also remove some outliers, namely those where the tip was larger than the fare cost
raw_data = raw_data[(raw_data['tip_amount'] <= raw_data['fare_amount'])]

# we remove trips with very large fare cost
raw_data = raw_data[((raw_data['fare_amount'] >=2) & (raw_data['fare_amount'] < 200))]

# we drop variables that include the target variable in it, namely the total_amount
clean_data = raw_data.drop(['total_amount'], axis=1)

# release memory occupied by raw_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
del raw_data
gc.collect()

# print the number of trips left in the dataset
print("There are " + str(len(clean_data)) + " observations in the dataset.")
print("There are " + str(len(clean_data.columns)) + " variables in the dataset.")

plt.hist(clean_data.tip_amount.values, 16, histtype='bar', facecolor='g')

# Histogram
hist_income = (
    px.histogram(clean_data, x='income')
    .update_layout(
        title='Histogram for Tip Data',
        title_x=0.5,
        bargap=0,  # Adjust this value to control the space between bars
        xaxis=dict(showgrid=True, layer='above traces'),
        yaxis=dict(showgrid=True, layer='above traces')
    )
    .update_traces(marker=dict(line=dict(color='black', width=2)))  # Outline color and width
)

print("Minimum amount value is ", np.min(clean_data.tip_amount.values))
print("Maximum amount value is ", np.max(clean_data.tip_amount.values))
print("90% of the trips have a tip amount less or equal than ", np.percentile(clean_data.tip_amount.values, 90))

# ========================== Train / Test Split ========================== #



# ========================== Confusion Matrix ========================== #

# Generate the confusion matrix


# ========================== Visualization ========================== #



# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Taxi Predictions', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%203%20-%20Classification/taxi_predict.py',
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
                    
                )
            ]
        ),
        html.Div(
            className='matrix',
            children=[
                
            ]
        ),
    ]
),
])

# if __name__ == '__main__':
#     app.run_server(debug=
#                    True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/yellow_tripdata.csv'
# data_path = os.path.join(script_dir, updated_path)
# raw_data.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #