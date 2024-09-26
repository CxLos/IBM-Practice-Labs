
# =============================== Imports ============================= #

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import itertools
import pandas as pd
import pylab as pl
import numpy as np
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

# ============================= Load Data ============================= #

# cell_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/cell_samples.csv'
file_path = os.path.join(script_dir, data_path)
cell_df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Exploration ========================== #

# print(cell_df.head())
# print('Columns:', cell_df.columns)
# print('Dtypes: \n', cell_df.dtypes)
# print('Class value counts: \n', cell_df['Class'].value_counts())
# print('Info:', cell_df.info())  
# print('Description:', cell_df.describe())
# print('Shape:', cell_df.shape)
# print('Size:', cell_df.size)

# ========================== DataFrame Table ========================== #

fig_head = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(cell_df.columns),
        fill_color='paleturquoise',
        align='left',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[cell_df[col] for col in cell_df.columns],
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

# ========================== Benign vs. Malignant ========================== #

# Create scatter plots for both classes
fig = px.scatter(
    cell_df[cell_df['Class'] == 4][0:50], # set to malignant
    x='Clump', #
    y='UnifSize',
    title='Scatter Plot of Clump vs UnifSize',
)

# Add scatter plot for malignant class with custom color and outline
fig.add_scatter(
    x=cell_df[cell_df['Class'] == 4][0:50]['Clump'], #
    y=cell_df[cell_df['Class'] == 4][0:50]['UnifSize'],
    mode='markers',
    marker=dict(color='dodgerblue', line=dict(color='Black', width=1)),
    name='malignant'
)

# Add scatter plot for benign class with custom color and outline
fig.add_scatter(
    x=cell_df[cell_df['Class'] == 2][0:50]['Clump'],
    y=cell_df[cell_df['Class'] == 2][0:50]['UnifSize'],
    mode='markers',
    marker=dict(color='Yellow', line=dict(color='Black', width=1)),
    name='benign'
)

fig.update_layout(
    xaxis_title='Clump Thickness',
    yaxis_title='Uniformity of Cell Size',
    title_x=0.5
    # legend_title='Class',
    # legend=dict(
    #     yanchor='top',
    #     y=0.99,
    #     xanchor='left',
    #     x=0.01
    # )
)

# Create a table using Plotly
# fig_head = go.Figure(data=[go.Table(
#     header=dict(values=list(cell_df.columns),
#                 fill_color='paleturquoise',
#                 align='left'),
#     cells=dict(values=[cell_df[col] for col in cell_df.columns],
#                fill_color='lavender',
#                align='left'))
# ])

# ========================== Data Pre Processing ========================== #

# Drop non numerical values
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

# Feature selection
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
# print(X[0:5])

# Target selection / Dependend variable
y = np.asarray(cell_df['Class'])
# print(y[0:5])

# ========================== Train / Test Split ========================== #

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

# ========================== SVM Model ========================== #

# Initialize the SVM model Non-Linear
clf = svm.SVC(kernel='rbf') #kernel='rbf' defines the kernel type to be used in the algorithm which is Radial Basis Function, used for non-linear classification

# Fit the model
clf.fit(X_train, y_train) 

# Predict
yhat = clf.predict(X_test)
# print(yhat [0:5])

#  LINEAR KERNEL
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)
print(yhat2 [0:5])
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))

# ========================== Evaluation ========================== #

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, 
                              yhat, 
                              labels=[2,4]) # 2: Benign, 4: Malignant
np.set_printoptions(precision=2)

# print ('Classification Report: \n', classification_report(y_test, yhat))

# f1_score
f1 = f1_score(y_test, yhat, average='weighted')
# print('F1 Score: ', f1)
# 0.9639038982104676

# jaccard_score
j_score = jaccard_score(y_test, yhat, pos_label=2)  
# print('Jaccard Score: ', j_score)
# 0.94444444444444444

# ========================== Data Visualization ========================== #

# Create the Heatmap
fig2 = go.Figure(data=go.Heatmap(
                    z=cnf_matrix,
                    x=['Benign(2)', 'Malignant(4)'],
                    y=['Benign(2)', 'Malignant(4)'],
                    colorscale='Blues',
                    showscale=True)
                )

# Update for better readability
fig2.update_layout(
    title='Confusion Matrix',
    xaxis_title='Predicted Value',
    yaxis_title='True Value',
    title_x=0.5
)

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Support Vector Machine (SVM)', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%203%20-%20Classification/svm.py',
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
                    children='Cell Data Table'
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
                  figure=fig
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  figure=fig2
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

# updated_path = 'data/cell_samples.csv'
# data_path = os.path.join(script_dir, updated_path)
# cell_df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #

# pip install dash --upgrade
# pip install dash-core-components --upgrade
# pip install dash-html-components --upgrade
# pip install dash-renderer --upgrade

# ========================================================================== #