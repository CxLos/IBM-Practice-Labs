
# ========================== Imports ========================== #

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

# ========================== Load Data ========================== #

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

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

#  ========================== Train/Test Split ========================== #

# Generate an array of booleans of the same length as the dataframe
# with 80% of the values being True
msk = np.random.rand(len(df)) < 0.8
# This uses the boolean mask msk to select rows from the DataFrame cdf where the mask is True. This results in approximately 80% of the rows being selected for the training dataset.
train = cdf[msk]
# This uses the negation of the boolean mask msk (i.e., ~msk) to select rows from the DataFrame cdf where the mask is False. This results in the remaining 20% of the rows being selected for the test dataset.
test = cdf[~msk]

# Initialize the linear regression model
regr = linear_model.LinearRegression()

# Train the model using the training sets
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# train_x = np.asanyarray(train[['FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']]) # Linear Way
# train_y = np.asanyarray(train['CO2EMISSIONS'])# Multi Way

# Converting test data to numpy arrays
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
# test_x = np.asanyarray(test[['FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]) #Linear Way
test_y = np.asanyarray(test[['CO2EMISSIONS']])
# test_y = np.asanyarray(test['CO2EMISSIONS']) # Multi Way

# fit the model using the training dataset
regr.fit (train_x, train_y)

# Flatten train_x to ensure it is a 1D array for plotly
train_x_flat = train_x.flatten()

# Predicting CO2 emissions using the test data
y_hat = regr.predict(test_x)

# Retrieve the coefficients and intercept
coefficients = regr.coef_
intercept = regr.intercept_
r2= r2_score(test_y, y_hat)
variance = regr.score(test_x, test_y)
mse= np.mean((y_hat - test_y) ** 2)

# print ('Coefficients: ', coefficients)
# print ('Intercept: ', intercept)
# print('R^2 score: ', r2)

# Calculating and printing the Mean Squared Error (MSE)
# '%' is the format specifier that indicates where the value should be placed in the string. '.2f' is the format specifier that indicates the value should be formatted as a fixed-point number with two decimal places.
# print("Mean Squared Error (MSE) : %.2f" % np.mean((y_hat - test_y) ** 2))

# Calculating and printing the Variance score using the feature matrix and true target values w/ the model making predictions internally
# print('Variance score: %.2f' % regr.score(test_x, test_y))

#  ========================== Data Visualization ========================== #
# 
# plot.update_traces(marker=dict(size=12))  # Example of updating traces

# Engine size vs C02 Emissiona

engine_scatter = px.scatter(
  cdf, 
  x='ENGINESIZE', 
  y='CO2EMISSIONS', 
  title='Engine Size vs C02 Emissions').update_layout(
    title='Engine Size vs C02 Emissions',
    title_x=0.5
  )

# Scatter Plot of Engine Size vs C02 Emissions using train dataset
engine_tt = px.scatter(
  train, 
  x='ENGINESIZE', 
  y='CO2EMISSIONS', 
  title='Engine Size vs C02 Emissions (Train Dataset)'
  ).update_layout(
    title='Engine Size vs C02 Emissions (Train Dataset)',
    title_x=0.5
  )

# Scatter Plot of Fuel Consumption Combo vs C02 Emissions using train dataset
fuel_combo = px.scatter(
    train,
    x='FUELCONSUMPTION_COMB',
    y='CO2EMISSIONS',
    title='Fuel Consumption Comb vs CO2 Emissions (Train Dataset)'
).update_layout(
    title='Fuel Consumption Comb vs CO2 Emissions (Train Dataset)',
    title_x=0.5
).add_trace(go.Scatter(
    # x=train_x_flat, #Multi way
    # x=train_x,
    x=train['FUELCONSUMPTION_COMB'], #Linear way
    # y=coefficients[0][0] * train_x_flat + intercept[0],
    # y=coefficients[1] * train_x_flat + intercept, # Multi way
    y=coefficients[0][2] * train['FUELCONSUMPTION_COMB'] + intercept[0], #Linear Way
    mode='lines',
    name='Regression Line',
    line=dict(color='red')
))

# Multiple linear regression using FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY 

x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit (x, y)

print ('Coefficients: ', regr.coef_)

y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))

# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# =========================== Dash App =========================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Multiple Linear Regression', 
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
                #   figure=engine_scatter
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    # figure=engine_tt
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
                    figure=fuel_combo
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

# -------------------------------------- Export Database -------------------------------------- #

# updated_path = 'data/fuel_consumption_C02.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ---------------------------------------------------------------------------------------------