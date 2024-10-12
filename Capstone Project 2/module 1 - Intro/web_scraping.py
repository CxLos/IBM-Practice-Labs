
# =============================== Imports ============================= #


import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import datetime
import requests
import pandas as pd
import pylab as pl
import numpy as np
import unicodedata
from bs4 import BeautifulSoup
import sys
import re
import os
import dash
from dash import dcc, html

# ============================= Load Data ============================= #

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)

static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

response = requests.get(static_url)
# print(response.content)

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/spacex.csv'
file_path = os.path.join(script_dir, data_path)
# df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Pre Processing ========================== #



# ========================== Web Scraping ========================== #

def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    
    
    # Use BeautifulSoup() to create a BeautifulSoup object from a response text content
soup = BeautifulSoup(response.content, "html.parser")

# Obtain Title
# print(soup.title)

# Use the find_all function in the BeautifulSoup object, with element type `table`
# Assign the result to a list called `html_tables`
html_tables = soup.find_all("table")

# Let's print the third table and check its content
first_launch_table = html_tables[2]
# print(first_launch_table)

# Next, we just need to iterate through the `<th>` elements and apply the provided `extract_column_from_header()` to extract column name one by one
column_names = []

# Apply find_all() function with `th` element on first_launch_table
# Iterate each th element and apply the provided extract_column_from_header() to get a column name
# Append the Non-empty column name (`if name is not None and len(name) > 0`) into a list called column_names

for header in first_launch_table.find_all('th'):
    name = extract_column_from_header(header)
    if name is not None and len(name) > 0:
        column_names.append(name)

launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column
del launch_dict['Date and time ( )']

# Let's initiate the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch Site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Add some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]

extracted_row = 0

#Extract each table 
for table_number, table in enumerate(soup.find_all('table', "wikitable plainrowheaders collapsible")):
    # Get table row
    for rows in table.find_all("tr"):
        # Check to see if first table heading is a number corresponding to launch a number
        if rows.th:
            if rows.th.string:
                flight_number = rows.th.string.strip()
                flag = flight_number.isdigit()
        else:
            flag = False
        # Get table element
        row = rows.find_all('td')
        # If it is a number, save cells in a dictionary
        if flag:
            extracted_row += 1
            # Flight Number value
            launch_dict['Flight No.'].append(flight_number)
            
            datatimelist = date_time(row[0])
            
            # Date value
            date = datatimelist[0].strip(',')
            launch_dict['Date'].append(date)
            
            # Time value
            time = datatimelist[1]
            launch_dict['Time'].append(time)
              
            # Booster version
            bv = booster_version(row[1])
            if not bv:
                bv = row[1].a.string
            launch_dict['Version Booster'].append(bv)
            
            # Launch Site
            launch_site = row[2].a.string
            launch_dict['Launch Site'].append(launch_site)
            
            # Payload
            payload = row[3].a.string
            launch_dict['Payload'].append(payload)
            
            # Payload Mass
            payload_mass = get_mass(row[4])
            launch_dict['Payload mass'].append(payload_mass)
            
            # Orbit
            orbit = row[5].a.string
            launch_dict['Orbit'].append(orbit)
            
            # Customer
            if row[6].a is not None:
                customer = row[6].a.string
            else:
                customer = None
            launch_dict['Customer'].append(customer)
            
            # Launch outcome
            launch_outcome = list(row[7].strings)[0]
            launch_dict['Launch outcome'].append(launch_outcome)
            
            # Booster landing
            booster_landing = landing_status(row[8])
            launch_dict['Booster landing'].append(booster_landing)

df= pd.DataFrame({ key:pd.Series(value) for key, value in launch_dict.items() })
# print(df.head())

# Count of 'Launche Site' from CCAFS, SLC-40
df_ccafs = df[df['Launch Site'] == 'CCAFS']
print('Launches from CCAFS: ', len(df_ccafs))

# site_count = df['Launch Site'].value_counts()
# print('Launch Site Value Counts: \n',site_count)

# Value counts of Launch Outcome at CCSFS SLC 40
launch_outcome_ccafs = df_ccafs['Launch outcome'].value_counts()
success_rate_ccafs = launch_outcome_ccafs['Success'] / launch_outcome_ccafs.sum() * 100
print(f'Success Rate: {success_rate_ccafs:.1f}%')

# Value counts of Orbit
df_orbit_gto = df[df['Orbit'] == 'GTO']
print('Orbit GTO:', len(df_orbit_gto))

# orbit_outcome = df['Orbit'].value_counts()
# print('Value Count Orbit Outcome: \n', orbit_outcome)

# Value counts of Landing Outcome
df_booster_landing_failures = df[df['Booster landing'] == 'Failure']
print('Booster Landing Failures:', len(df_booster_landing_failures))

# landing_outcome = df['Booster landing'].value_counts()
# print('Value Count Landing Outcome: \n', landing_outcome)

# ========================== DataFrame Table ========================== #

df_table = go.Figure(data=[go.Table(
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

df_table.update_layout(
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
                    figure=df_table
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

# updated_path = 'data/spacex.csv'
# data_path = os.path.join(script_dir, updated_path)
# data.to_csv(data_path, index=False)
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