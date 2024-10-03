
# =============================== Imports ============================= #

from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/cust_segmentation.csv'
file_path = os.path.join(script_dir, data_path)
cust_df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Pre Processing ========================== #

# Drop non numerical values
df = cust_df.drop('Address', axis=1)



# ========================== K - Means ========================== #

# Random seed:
np.random.seed(0) # .seed(0) means that the random numbers will be the same each time you run the code.

# make blobs is a function that generates isotropic Gaussian blobs for clustering.
X, y = make_blobs(n_samples=5000, # The total number of points equally divided among clusters.
                #   centers=[[4,4], [-2, -1], [2, -3], [1, 1]], # The number of centers to generate, or the fixed center locations.
                  centers=[[4,4], [-2, -1], [2, -3]],
                  cluster_std=0.9) # The standard deviation of the clusters.

# Scatter Plot:
scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers')

# Initialize KMeans:
# k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

# Fit the model:
k_means.fit(X)

# Get the labels:
k_means_labels = k_means.labels_

# Get the cluster centers:
k_means_cluster_centers = k_means.cluster_centers_

# Initialize the plot with the specified dimensions.
cluster = go.Figure()

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
# colors = ['red', 'orange', 'green', 'dodgerblue']
colors = ['red', 'green', 'dodgerblue']
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
# for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3]])), colors):
    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Extract the RGB values from the col variable
    rgb = mcolors.to_rgb(col)
    
    # Generate a lighter shade for the data points
    lighter_rgb = tuple(min(1, x + 0.5) for x in rgb)
    
    # Convert the lighter RGB values back to a color string
    lighter_col = mcolors.to_hex(lighter_rgb)
    
    # Plots the datapoints with the lighter shade of the centroid color.
    cluster.add_trace(go.Scatter(
        x=X[my_members, 0],
        y=X[my_members, 1],
        mode='markers',
        marker=dict(color=lighter_col, size=6, line=dict(color='black', width=0.5)),
        name=f'Cluster {k}'
    ))
    
    # Plots the centroids with specified color, but with a darker outline
    cluster.add_trace(go.Scatter(
        x=[cluster_center[0]],
        y=[cluster_center[1]],
        mode='markers',
        marker=dict(color=col, size=12, line=dict(color='black', width=2)),
        name=f'Centroid {k}'
    ))

# Title of the plot
cluster.update_layout(title='KMeans', title_x=0.5)

# Remove x-axis and y-axis ticks
cluster.update_xaxes(showticklabels=False)
cluster.update_yaxes(showticklabels=False)

# ========================== Train / Test Split ========================== #

# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

# ========================== Modeling ========================== #

# Feature selection
x = df.values[:,1:] # all rows, all columns except the first
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(x)
labels = k_means.labels_

# Assigning the labels to each row in the dataframe. 
df["Clus_km"] = labels

# Check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

# ========================== Data Visualization ========================== #

# distribution of customers based on their age and income:

# Calculate the area for the scatter plot
area = np.pi * (x[:, 1])**2

# Create a Plotly figure
age_income = go.Figure()

# Add a scatter plot to the figure
age_income.add_trace(go.Scatter(
    x=x[:, 0],
    y=x[:, 3],
    mode='markers',
    marker=dict(
        size=area,
        color=k_means_labels.astype(float),
        opacity=0.5,
        colorscale='Viridis',
        showscale=True
    )
))

# Update the layout of the figure
age_income.update_layout(
    xaxis_title='Age',
    yaxis_title='Income',
    title='Age vs Income Scatter Plot',
    title_x=0.5
)

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
        
        html.H1('K-Means Clustering', 
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
                    children='Customer Segmentation Data Table'
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
                  figure=cluster
                # figure={'data': [scatter]}
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    # figure=cluster
                    figure=age_income
                    # figure={'data': [cluster]}
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

# updated_path = 'data/cust_segmentation.csv'
# data_path = os.path.join(script_dir, updated_path)
# cust_df.to_csv(data_path, index=False)
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