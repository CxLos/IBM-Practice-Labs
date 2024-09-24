
# =============================== Imports ============================= #

from __future__ import print_function
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from snapml import DecisionTreeClassifier
import pandas as pd
import pylab as pl
import numpy as np
import time
import os
import dash
from dash import dcc, html
from dash.development.base_component import Component

# ========================== Load Data ==========================

raw_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv')

# current_dir = os.getcwd()
# script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = 'data/credit_card.csv'
# file_path = os.path.join(script_dir, data_path)
# raw_data = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Exploration ========================== #

# Preview Data
# print(raw_data.head())

# Summarize the data
# print(df.describe())
# print(df.info())
# print(df.columns)
# print(df.shape)
# print(df.dtypes)

# print("There are " + str(len(df)) + " observations in the credit card fraud dataset.")
# print("There are " + str(len(df.columns)) + " variables in the dataset.")

# In real life, there will be much larger datasets, so lets inflate the dataset to 10 times its original size
n_replicas = 10

# inflate the original dataset
# big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)

# print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
# print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")

# display first rows in the new dataset
# print(big_raw_data.head())

# get the set of distinct classes
labels = raw_data.Class.unique()
# labels = big_raw_data.Class.unique()
# print(labels)

# get the count of each class
sizes = raw_data.Class.value_counts().values
# sizes = big_raw_data.Class.value_counts().values
# print(sizes)

# print("Minimum amount value is ", np.min(raw_data.Amount.values))
# print("Maximum amount value is ", np.max(raw_data.Amount.values))
# print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))

# ========================== Train/ Test Split ========================== #

# data preprocessing such as scaling/normalization is typically useful for 
# linear models to accelerate the training convergence

# standardize features by removing the mean and scaling to unit variance
# standardization basically helps to normalizes the data within a particular range
# .fittransform computes the mean and std dev for future use
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

# print the shape of the features matrix and the labels vector
# print('X.shape=', X.shape, 'y.shape=', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       
# print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
# print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

# ========================== Decision Tree ========================== #

# compute the sample weights to be used as input to the train routine so that 
# it takes into account the class imbalance present in this dataset
w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)

# train a Decision Tree Classifier using scikit-learn
t0 = time.time()
sklearn_time = time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

# ========================== Data Visualization ========================== #

# Class Pie Chart
fig = go.Figure(
    data=[go.Pie(
        labels=labels, 
        values=sizes, 
        textinfo='label+percent', 
        insidetextorientation='radial')
        ])

fig.update_layout(
    title_text='Target Variable Value Counts',
    title_x=0.5,)

# Histogram for cc transaction amounts
hist_transaction = (
    px.histogram(raw_data, x='Amount', nbins=30)
    .update_layout(
        title='Histogram for Transaction Amounts',
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
        
        html.H1('Decision Tree / SVM', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%203%20-%20Classification/decision_tree.py',
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
                    figure=fig
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=hist_transaction
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

# updated_path = 'data/credit_card.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #

# pip install dash --upgrade
# pip install dash-core-components --upgrade
# pip install dash-html-components --upgrade
# pip install dash-renderer --upgrade

# ========================================================================== #