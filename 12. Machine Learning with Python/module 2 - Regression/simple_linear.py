# ========================== Imports ==========================

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

#  ========================== Data Visualization ========================== #

# features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Histograms
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz_hist_cylinder = px.histogram(viz, x='CYLINDERS')
viz_hist_enginesize = px.histogram(viz, x='ENGINESIZE')
viz_hist_co2emissions = px.histogram(viz, x='CO2EMISSIONS')
viz_hist_fuelconsumption = px.histogram(viz, x='FUELCONSUMPTION_COMB')

# Scatter plots
viz_scatter_fuel = go.Figure(data=go.Scatter(
    x=cdf['FUELCONSUMPTION_COMB'],
    y=cdf['CO2EMISSIONS'],
    mode='markers'
))

viz_scatter_engine = go.Figure(data=go.Scatter(
    x=cdf['ENGINESIZE'],
    y=cdf['CO2EMISSIONS'],
    mode='markers'
))

viz_scatter_cylinder = go.Figure(data=go.Scatter(
    x=cdf['CYLINDERS'],
    y=cdf['CO2EMISSIONS'],
    mode='markers'
))

# Enable grid lines
# for fig in [viz_hist_cylinder, viz_hist_enginesize, viz_hist_co2emissions, viz_hist_fuelconsumption]:
#     fig.update_layout(
#         xaxis=dict(showgrid=True, layer='above traces'),
#         yaxis=dict(showgrid=True, layer='above traces'),
#         bargap=0.2  # Adjust this value to control the space between bars
#     )
#     fig.update_traces(marker=dict(line=dict(color='black', width=1))) 

#  ========================== Train/Test Split ========================== #

# 80% of the data is used for training and 20% for testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# -------------------------------------------------------------------------------

# Linear regression model
regr = linear_model.LinearRegression()

# Train the model using the training sets FUELCONSUMPTION_COMB
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Fit the model
regr.fit(train_x, train_y)

# predict the CO2 emissions of the test data
test_y_ = regr.predict(test_x)

# Flatten train_x to ensure it is a 1D array for plotly
train_x_flat = train_x.flatten()

# The coefficients
coefficients = regr.coef_
intercept = regr.intercept_

# print('Coefficient: ', coefficients)
# print('Intercept: ', intercept)
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y , test_y_) )

# Initialize the figure for the engine size vs CO2 emissions plot
msk_engine = go.Figure()

# Add scatter points for engine size vs CO2 emissions
msk_engine.add_trace(go.Scatter(
    x=train['ENGINESIZE'],
    y=train['CO2EMISSIONS'],
    mode='markers',
    name='Data points',
    marker=dict(color='blue')
))

# Add regression line
msk_engine.add_trace(go.Scatter(
    x=train_x_flat,  # X-values for the regression line
    y=coefficients[0][0] * train_x_flat + intercept[0],  # Y-values from the regression equation
    mode='lines',
    name='Regression Line',
    line=dict(color='red')
))

# Set x-axis and y-axis labels
msk_engine.update_layout(
    xaxis_title="Engine Size",
    yaxis_title="Emission",
    annotations=[
        dict(
            x=6.8,  # Position the annotation within the data range
            y=170,  # Position the annotation within the data range
            xref='x',
            yref='y',
            text=f'Coefficient: {regr.coef_[0][0]:.2f}, Intercept: {regr.intercept_[0]:.2f}',
            showarrow=False,
            font=dict(size=15)
        )
    ]
)

# -------------------------------------------------------------------------------

# Linear regression model
regr = linear_model.LinearRegression()

# Initialize the figure for the FUELCONSUMPTION_COMB vs CO2 emissions plot
msk_fuel = go.Figure()

# Train the model using the training sets FUELCONSUMPTION_COMB
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Fit the model
regr.fit(train_x, train_y)

# predict the CO2 emissions of the test data
test_y_ = regr.predict(test_x)

# Flatten train_x to ensure it is a 1D array for plotly
train_x_flat = train_x.flatten()

# Add scatter points for FUELCONSUMPTION_COMB vs CO2 emissions
msk_fuel.add_trace(go.Scatter(
    x=train['FUELCONSUMPTION_COMB'],
    y=train['CO2EMISSIONS'],
    mode='markers',
    name='Data points',
    marker=dict(color='blue')
))

# Add regression line 
msk_fuel.add_trace(go.Scatter(
    x=train['FUELCONSUMPTION_COMB'],
    y=coefficients[0][0] * train_x_flat + intercept[0],
    mode='lines',
    name='Regression Line',
    line=dict(color='red')
))

# Set x-axis and y-axis labels
msk_fuel.update_layout(
    title='Engine Size Regression Line',
    xaxis_title="Fuel Consumption",
    yaxis_title="Emission"
    # ,annotations=[
    #     dict(
    #         x=6.8,  # Position the annotation within the data range
    #         y=170,  # Position the annotation within the data range
    #         xref='x',
    #         yref='y',
    #         text=f'Coefficient: {regr.coef_[0][0]:.2f}, Intercept: {regr.intercept_[0]:.2f}',
    #         showarrow=False,
    #         font=dict(size=15)
    #     )
    # ]
)

# Predit Mean Absolute Error for fuel consumption model
predictions = regr.predict(test_x)
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))

# =========================== Dash App =========================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Simple Linear Regression', 
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
                  figure=viz_hist_cylinder
                    .update_layout(
                      title='Histogram for CYLINDERS',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True, layer='above traces'),
                      yaxis=dict(showgrid=True, layer='above traces')
                     
                    )
                    # Outline color and width
                    .update_traces(marker=dict(line=dict(color='black', width=2)))  
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
                      title='Histogram for ENGINESIZE',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                     # Outline color and width
                    .update_traces(marker=dict(color='red',line=dict(color='black', width=1))) 
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
                  #  Histogram for 'CO2EMISSIONS'                  
                  figure=viz_hist_co2emissions
                  .update_layout(
                      title='Histogram for CO2EMISSIONS',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                    #  Outline color and width
                    .update_traces(marker=dict(color='orange',line=dict(color='black', width=1)))
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
                      title='Histogram for FUELCONSUMPTION',
                      title_x=0.5,
                      bargap=0,  # Adjust this value to control the space between bars
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True)
                    )
                    # Outline color and width
                    .update_traces(marker=dict(color='lightblue',line=dict(color='black', width=1)))  
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
                                  
                  figure=viz_scatter_fuel
                  .update_layout(
                    title='Fuel Consumption vs CO2 Emissions',
                    title_x=0.5,
                    xaxis_title='Engine size',
                    yaxis_title='Emission'
                )
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  figure=viz_scatter_engine
                  .update_layout(
                    title='Engine Size vs CO2 Emissions',
                    title_x=0.5,
                    xaxis_title='Fuel Consumption',
                    yaxis_title='Emission'
                    )
                    .update_traces(marker=dict(color='red',line=dict(color='black', width=1)))
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
                  figure=viz_scatter_cylinder
                  .update_layout(
                    title='Cylinders vs CO2 Emissions',
                    title_x=0.5,
                    xaxis_title='Cylinders',
                    yaxis_title='Emission'
                    ).update_traces(marker=dict(color='green',line=dict(color='black', width=1)))
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  figure=msk_engine
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
                  figure=msk_fuel
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

# ----------------------- Export Database ------------------------------- #

# updated_path = 'data/fuel_consumption_C02.csv'
# data_path = os.path.join(script_dir, updated_path)
# df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# -------------------------------------------------------------------------------