
# =============================== Imports ============================= #

import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import csv, sqlite3
import itertools
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html

# ============================= Load Data ============================= #

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/spacex_part_1.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# SQL
con = sqlite3.connect("ibm_practice_labs.db")
cur = con.cursor()
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

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

# 1. Display the names of the unique launch sites in the space mission
# query = pd.read_sql_query("""
#   SELECT DISTINCT Launch_Site FROM SPACEXTBL
#   """, con)

#     Launch_Site
# 0   CCAFS LC-40
# 1   VAFB SLC-4E
# 2    KSC LC-39A
# 3  CCAFS SLC-40

# 2. Display 5 records where launch sites begin with the string 'CCA' 
# query = pd.read_sql_query("""
#   SELECT Date, "Time (UTC)", Launch_Site
#   FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%'
#   Limit 5
#   """, con)

# print(query)

#          Date Time (UTC)  Launch_Site
# 0  2010-06-04   18:45:00  CCAFS LC-40
# 1  2010-12-08   15:43:00  CCAFS LC-40
# 2  2012-05-22    7:44:00  CCAFS LC-40
# 3  2012-10-08    0:35:00  CCAFS LC-40
# 4  2013-03-01   15:10:00  CCAFS LC-40

# 3. Display the total payload mass carried by boosters launched by NASA (CRS)
# query = pd.read_sql_query("""
#   SELECT SUM(PAYLOAD_MASS__KG_) 
#   FROM SPACEXTBL 
#   WHERE Customer='NASA (CRS)'
#   """, con)

# print(query)

#    SUM(PAYLOAD_MASS__KG_)
# 0                   45596

# 4. Display average payload mass carried by booster version F9 v1.1
# query = pd.read_sql_query("""
#   SELECT AVG(PAYLOAD_MASS__KG_)
#   FROM SPACEXTBL
#   WHERE Booster_Version='F9 v1.1'
#   """, con)

# print(query)

#    AVG(PAYLOAD_MASS__KG_)
# 0                  2928.4

# 5. List the date when the first succesful landing outcome in ground pad was acheived.
# query = pd.read_sql_query("""
#   SELECT MIN(Date)
#   FROM SPACEXTBL
#   WHERE Landing_Outcome='Success (ground pad)'
#   """, con)

# print(query)

#     MIN(Date)
# 0  2015-12-22

# 6. List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
# query = pd.read_sql_query("""
#   SELECT Booster_Version, Landing_Outcome
#   FROM SPACEXTBL
#   WHERE Landing_Outcome='Success (drone ship)'
#   AND PAYLOAD_MASS__KG_ BETWEEN 4000 AND 6000
#   """, con)

# print(query)

#   Booster_Version       Landing_Outcome
# 0     F9 FT B1022  Success (drone ship)
# 1     F9 FT B1026  Success (drone ship)
# 2  F9 FT  B1021.2  Success (drone ship)
# 3  F9 FT  B1031.2  Success (drone ship)

# 7. List the total number of successful and failure mission outcomes
# query = pd.read_sql_query("""
#   SELECT Count(Mission_Outcome)
#   FROM SPACEXTBL
#   WHERE Mission_Outcome='Success' OR Mission_Outcome='Failure'
#   """, con)

# print(query)

#    Count(Mission_Outcome)
# 0                      98

# 8. List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
# query = pd.read_sql_query("""
#   SELECT Booster_Version, MAX(PAYLOAD_MASS__KG_)
#   FROM SPACEXTBL
#   WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL)
#   """, con)

# print(query)

#   Booster_Version  MAX(PAYLOAD_MASS__KG_)
# 0   F9 B5 B1048.4                   15600

# 9. List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.
# query = pd.read_sql_query("""
#   SELECT strftime('%B', (Date)) as Month, Landing_Outcome, Booster_Version, Launch_Site
#   FROM SPACEXTBL
#   WHERE Landing_Outcome='Failure (drone ship)'
#   AND strftime('%Y', (Date))='2015'
#   """, con)

# print(query)

#   Month       Landing_Outcome Booster_Version  Launch_Site
# 0  None  Failure (drone ship)   F9 v1.1 B1012  CCAFS LC-40
# 1  None  Failure (drone ship)   F9 v1.1 B1015  CCAFS LC-40

# 10. Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.
# query = pd.read_sql_query("""
#   SELECT Landing_Outcome, COUNT(Landing_Outcome) as Count
#   FROM SPACEXTBL
#   WHERE Date BETWEEN '2010-06-04' AND '2017-03-20'
#   GROUP BY Landing_Outcome 
#   ORDER BY Count DESC
#   """, con)

# print(query)

#           Landing_Outcome  Count
# 0              No attempt     10
# 1    Success (drone ship)      5
# 2    Failure (drone ship)      5
# 3    Success (ground pad)      3
# 4      Controlled (ocean)      3
# 5    Uncontrolled (ocean)      2
# 6     Failure (parachute)      2
# 7  Precluded (drone ship)      1

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
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%205%20-%20Clustering/k_means_clustering.py',
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

# if __name__ == '__main__':
#     app.run_server(debug=
#                    True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/spacex_part_1.csv'
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