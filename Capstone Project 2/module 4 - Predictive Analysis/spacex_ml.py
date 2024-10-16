
# =============================== Imports ============================= #

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
df2=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/spacex_ml.csv'
data_path2 = 'data/spacex_ml2.csv'
file_path = os.path.join(script_dir, data_path)
file_path2 = os.path.join(script_dir, data_path2)
df = pd.read_csv(file_path)
X = pd.read_csv(file_path2)

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

# ========================== Confusion Matrix ========================== #

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 

# ========================== Questions ========================== #

# 1. Create a NumPy array from the column <code>Class</code> in <code>data</code>, by applying the method <code>to_numpy()</code>  then
# assign it  to the variable <code>Y</code>,make sure the output is a  Pandas series (only one bracket df\['name of  column']).

Y = df['Class'].to_numpy()

# 2. Standardize the data in <code>X</code> then reassign it to the variable  <code>X</code> using the transform provided below:
transform = preprocessing.StandardScaler()
X = transform.fit(X).transform(X)

# 3. Split the data into training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 4. Create a logistic regression object  then create a  GridSearchCV object  <code>logreg_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.

LR = LogisticRegression()
LR.fit(X_train, Y_train)

GridSearchCV(
    estimator=LR,
    param_grid={'C': np.logspace(-5, 8, 15), 'penalty': ['l1', 'l2']},
    cv=10
)

parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}

# ========================== Questions ========================== #



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
                  # figure=fig_0
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  # figure=fig_1
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
                  # figure=fig_2
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                  # figure=fig_3
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

# updated_path = 'data/spacex_ml2.csv'
# data_path = os.path.join(script_dir, updated_path)
# df2.to_csv(data_path, index=False)
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