
# =============================== Imports ============================= #

import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import matplotlib.pyplot as plt
import csv, sqlite3
import itertools
import seaborn as sns
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html

# ============================= Load Data ============================= #

# df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/spacex_part_2.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Pre Processing ========================== #

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.columns)
# print(df.dtypes)

# ========================== Columns ========================== #

#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   Date               101 non-null    object
#  1   Time (UTC)         101 non-null    object
#  2   Booster_Version    101 non-null    object
#  3   Launch_Site        101 non-null    object
#  4   Payload            101 non-null    object
#  5   PAYLOAD_MASS__KG_  101 non-null    int64
#  6   Orbit              101 non-null    object
#  7   Customer           101 non-null    object
#  8   Mission_Outcome    101 non-null    object
#  9   Landing_Outcome    101 non-null    object

# ========================== Questions ========================== #

# First, let's try to see how the `FlightNumber` (indicating the continuous launch attempts.) and `Payload` variables would affect the launch outcome.

# We can plot out the <code>FlightNumber</code> vs. <code>PayloadMass</code>and overlay the outcome of the launch. We see that as the flight number increases, the first stage is more likely to land successfully. The payload mass is also important; it seems the more massive the payload, the less likely the first stage will return.

fig_0 = px.scatter(
    df,
    x="FlightNumber",
    y="PayloadMass",
    color="Class",
    title="Flight Number vs Payload Mass",
    labels={"FlightNumber": "Flight Number", "PayloadMass": "Payload Mass (kg)"}
)

# Update the layout of the figure
fig_0.update_layout(
    xaxis_title="Flight Number",
    yaxis_title="Payload Mass (kg)",
    title_font_size=20,
    xaxis_title_font_size=20,
    yaxis_title_font_size=20
)

# sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
# plt.xlabel("Flight Number",fontsize=20)
# plt.ylabel("Pay load Mass (kg)",fontsize=20)
# plt.show()

# 1. Use the function catplot to plot FlightNumber vs LaunchSite, set the  parameter x  parameter to FlightNumber, set the y to Launch Site and set the parameter hue to 'class'
fig_1 = px.scatter(
    df, 
    x='FlightNumber', 
    y='LaunchSite', 
    color='Class', 
    title='Flight Number vs Launch Site')

# 2. Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value

fig_2 = px.scatter(
    df,
    x='PayloadMass',
    y='LaunchSite',
    color='Class',
    title='Payload Mass vs Launch Site'
)

# 3. Let's create a `bar chart` for the sucess rate of each orbit

# 4. Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value

# 5. Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value

# 6. plot a line chart with x axis to be <code>Year</code> and y axis to be average success rate, to get the average launch success trend. 

# A function to Extract years from the date 
# year=[]
# def Extract_year(date):
#     for i in df["Date"]:
#         year.append(i.split("-")[0])
#     return year
# # Plot a line chart with x axis to be the extracted year and y axis to be the success rate
# df["Year"]=Extract_year(df["Date"])
# df["Class"]=df["Class"].astype(int)
# suc_rate=df.groupby("Year")["Class"].mean().reset_index()

# features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
# features.head()

# 7. Use the function get_dummies and features dataframe to apply OneHotEncoder to the column Orbits, LaunchSite, LandingPad, and Serial. Assign the value to the variable features_one_hot, display the results using the method head. Your result dataframe must include all features including the encoded ones.

# 8. Now that our features_one_hot dataframe only contains numbers cast the entire dataframe to variable type float64

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

# ROW 1
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(
                  figure=fig_0
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  figure=fig_1
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
                  figure=fig_2
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

# updated_path = 'data/spacex_part_2.csv'
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