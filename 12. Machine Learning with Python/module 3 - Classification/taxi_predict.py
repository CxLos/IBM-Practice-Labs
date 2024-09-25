
# ================================ Imports ======================== #

import pandas as pd
import pylab as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as opt
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import itertools
import base64
import time
import os
import warnings
import gc, sys
from io import BytesIO
import dash
from dash import dcc, html
from dash.development.base_component import Component

# ================================ Data =========================== #

raw_data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/yellow_tripdata_2019-06.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/yellow_tripdata.csv'
file_path = os.path.join(script_dir, data_path)
# raw_data = pd.read_csv(file_path)

# ========================== Data Exploration ========================== #

# Preview Data
# print(churn_df.head())

# Summarize the data
# print("DF Shape:", churn_df.shape)
# print("DTypes: \n", churn_df.dtypes)
# print("Description: \n", churn_df.describe())
# print("Info:", churn_df.info())
# print("Columns:", churn_df.columns)

# ========================== Cleaning Data ========================== #

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
# print(clean_data.head())

# release memory occupied by raw_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
del raw_data
gc.collect()

# print the number of trips left in the dataset
# print("There are " + str(len(clean_data)) + " observations in the dataset.")
# print("There are " + str(len(clean_data.columns)) + " variables in the dataset.")

# print("Minimum amount value is ", np.min(clean_data.tip_amount.values))
# print("Maximum amount value is ", np.max(clean_data.tip_amount.values))
# print("90% of the trips have a tip amount less or equal than ", np.percentile(clean_data.tip_amount.values, 90))

# ========================== Pre Processing ========================== #

# convert to datetime
clean_data['tpep_dropoff_datetime'] = pd.to_datetime(clean_data['tpep_dropoff_datetime'])
clean_data['tpep_pickup_datetime'] = pd.to_datetime(clean_data['tpep_pickup_datetime'])

# extract pickup and dropoff hour
clean_data['pickup_hour'] = clean_data['tpep_pickup_datetime'].dt.hour
clean_data['dropoff_hour'] = clean_data['tpep_dropoff_datetime'].dt.hour

# extract pickup and dropoff day of week
clean_data['pickup_day'] = clean_data['tpep_pickup_datetime'].dt.weekday
clean_data['dropoff_day'] = clean_data['tpep_dropoff_datetime'].dt.weekday

# compute trip time in minutes
clean_data['trip_time'] = (clean_data['tpep_dropoff_datetime'] - clean_data['tpep_pickup_datetime']).dt.total_seconds() / 60

# reduce dataset size if needed
first_n_rows = 1000000
clean_data = clean_data.head(first_n_rows)

# drop the pickup and dropoff datetimes
clean_data = clean_data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

# some features are categorical, we need to encode them
# to encode them we use one-hot encoding from the Pandas package
get_dummy_col = ["VendorID","RatecodeID","store_and_fwd_flag","PULocationID", "DOLocationID","payment_type", "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
proc_data = pd.get_dummies(clean_data, columns = get_dummy_col)

# Histogram Tip Data
hist_trip = (
    px.histogram(clean_data, x='tip_amount', nbins=16)
    .update_layout(
        title='Histogram for Tip Data',
        title_x=0.5,
        bargap=0,  # Adjust this value to control the space between bars
        xaxis=dict(showgrid=True, layer='above traces'),
        yaxis=dict(showgrid=True, layer='above traces')
    )
    .update_traces(marker=dict(line=dict(color='black', width=2)))  # Outline color and width
)

# release memory occupied by clean_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
# del clean_data

# release memory occupied by clean_data as we do not need it anymore
# gc.collect()

# extract the labels from the dataframe
y = proc_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = proc_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

# print the shape of the features matrix and the labels vector
# print('X.shape=', X.shape, 'y.shape=', y.shape)

# ========================== Train / Test Split ========================== #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
# print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

# ======================= Decision Tree Regressor ======================== #

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeRegressor(max_depth=8, random_state=35)

dt.fit(X_train,y_train)

# Visualize Tree
fig, ax = plt.subplots(figsize=(20, 10))  # Set the width and height of the plot

# Set font size, feature name and max depth of the tree
features = X_train.columns.tolist()
tree.plot_tree(
  dt, # the data to be plotted
  ax=ax, # the axes to plot the data
  fontsize=10, 
  feature_names=features, 
  max_depth=8)

# Convert the Matplotlib figure to a base64-encoded string
buffer = BytesIO()
fig.savefig(buffer, format='png')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

# Save the plot to a file
fig_path = 'images/tree1.png'
img_path = os.path.join(script_dir, fig_path)
fig.savefig(img_path)

# train a Decision Tree Regressor using scikit-learn
t0 = time.time()
dt.fit(X_train, y_train)
sklearn_time = time.time()-t0
# print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

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
                    figure=hist_trip
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                html.Img(src=f'data:image/png;base64,{img_base64}')
            ]
        ),
    ]
),
# ROW 1
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
                dcc.Graph()
            ]
        ),
    ]
),
])

if __name__ == '__main__':
    app.run_server(debug=
                   True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/yellow_tripdata.csv'
# data_path = os.path.join(script_dir, updated_path)
# raw_data.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #