
# =============================== Imports ============================= #

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
# print(y[0:5]) # Display the first 5 rows

# Normalize Data
# .fit computes the mean and std dev for future use
# .transform applies the normalization to the data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X[0:5]) # Display the first 5 rows

# Train Test Split
# trrain 80% of the data and test 20%
# random_state is the seed for the random number generator
# random number generator is used to shuffle the data before splitting it
# random_state ensures that the splits that you generate are reproducible
# setting it to 4 means you'll get the same output every time you run your code
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

# Training
k = 4

#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# print(neigh)

# Predicting
yhat = neigh.predict(X_test)
# print("Predictions:", yhat[0:5])

# Accuracy Evaluation
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Finding the right value for k

# set the number of k's to test
Ks = 10
# np.zeros creates an array of zeros
# mean_acc will store the mean accuracy of each value of k
# std_acc will store the standard deviation of the accuracy of each value of k
# we will use these arrays to plot the model accuracy for different values of k
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    # print(mean_acc)

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

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

# Plot the model accuracy

# Create the plotly figure
fig = go.Figure()

# Add the accuracy line
fig.add_trace(go.Scatter(
    x=list(range(1, Ks)),
    y=mean_acc,
    mode='lines',
    name='Accuracy',
    line=dict(color='green')
))

# Add the +/- 1x std fill
fig.add_trace(go.Scatter(
    # Create list of Ks and reverse it to create the bottom of the fill
    # This will create a shape that is the mean +/- 1x std
    x=list(range(1, Ks)) + list(range(1, Ks))[::-1], # x, then x reversed
    # Create list of mean accuracy +/- 1x std
    # This will create a shape that is the mean +/- 1x std
    y=list(mean_acc - 1 * std_acc) + list((mean_acc + 1 * std_acc))[::-1],
    fill='toself', # Fill in the shape
    fillcolor='rgba(0, 255, 0, 0.1)', # Fill color
    line=dict(color='rgba(255, 255, 255, 0)'), # No line
    hoverinfo="skip", # No hover info
    showlegend=True, # Show in legend
    name='+/- 1xstd' # Name in legend
))

# Add the +/- 3x std fill
fig.add_trace(go.Scatter(
    x=list(range(1, Ks)) + list(range(1, Ks))[::-1],
    y=list(mean_acc - 3 * std_acc) + list((mean_acc + 3 * std_acc))[::-1],
    fill='toself',
    fillcolor='rgba(0, 255, 0, 0.1)',
    line=dict(color='rgba(255, 255, 255, 0)'),
    hoverinfo="skip",
    showlegend=True,
    name='+/- 3xstd'
))

# Update layout
fig.update_layout(
    title='Accuracy vs. Number of Neighbors (K)',
    xaxis_title='Number of Neighbors (K)',
    yaxis_title='Accuracy',
    legend_title='Legend',
    template='plotly_white',
    title_x=0.5
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
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%203%20-%20Classification/knn.py',
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
                  figure=fig
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

# updated_path = 'data/telecust1000.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #

# pip install dash --upgrade
# pip install dash-core-components --upgrade
# pip install dash-html-components --upgrade
# pip install dash-renderer --upgrade

# ========================================================================== #