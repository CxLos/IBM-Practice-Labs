
# =============================== Imports ============================= #

from sklearn.cluster import KMeans 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
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

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/australia_weather_data.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== Data Pre Processing ========================== #

# One-Hot Encoding
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

# Replace 'No' and 'Yes' with 0 and 1 so that we do not end up with 2 columns for target variable
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

# Drop date column as we do not need it
df_sydney_processed.drop('Date',axis=1,inplace=True)

# Convert all columns to float
df_sydney_processed = df_sydney_processed.astype(float)

# Feature set / x values
features = df_sydney_processed.drop(columns='RainTomorrow', 
                                    axis=1) # axis 1 means column-wise

Y = df_sydney_processed['RainTomorrow']

# ========================== 1. Linear Regression ========================== #

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

LinearReg = LinearRegression()

LinearReg.fit(x_train, y_train)

predictions = LinearReg.predict(x_test)

LinearRegression_MAE = metrics.mean_absolute_error(y_test, predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test, predictions)
LinearRegression_R2 = metrics.r2_score(y_test, predictions)

# print("Linear Regression Mean Absolute Error: ", LinearRegression_MAE)
# print("Linear Regression Mean Squared Error: ", LinearRegression_MSE)
# print("Linear Regression R2 Score: ", LinearRegression_R2)

# Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.
# Rename Column names as well
linear_report = pd.DataFrame(data=[["Linear Regression", LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]], columns=['Model', 'Mean Absolute Error', 'Mean Squared Error', 'R2 Score'])

# print(Report)

# ========================== 2. KNN ========================== #

KNN = KNeighborsClassifier(n_neighbors=4)

predictions = KNN.fit(x_train, y_train).predict(x_test)

KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

knn_report = pd.DataFrame(data=[["KNN", KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score]], columns=['Model', 'Accuracy Score', 'Jaccard Index', 'F1 Score'])

# print(report)

# ========================== 3. Decision Tree ========================== #

Tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)

Tree.fit(x_train, y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

dt_report = (pd.DataFrame(data=[["Decision Tree", Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score]], columns=['Model', 'Accuracy Score', 'Jaccard Index', 'F1 Score']))

# ========================== 4. Logistic Regression ========================== #

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Solver is used to specify the optimization algorithm meaning the algorithm that will be used to find the minimum value of the cost function
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train, y_train)

predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)

LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)

logistic_report = pd.DataFrame(data=[["Logistic Regression", LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score, LR_Log_Loss]], columns=['Model', 'Accuracy Score', 'Jaccard Index', 'F1 Score', 'Log Loss'])

# ========================== 5. SVM ========================== #

# SVM = svm.SVC(kernel='rbf')
SVM = svm.SVC(kernel='linear')
SVM.fit(x_train, y_train)

predictions = SVM.predict(x_test)

SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

svm_report = pd.DataFrame(data=[["SVM", SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]], columns=['Model', 'Accuracy Score', 'Jaccard Index', 'F1 Score'])

# ========================== Model Evaluation ========================== #

final_report = pd.concat([linear_report, knn_report, dt_report, logistic_report, svm_report], ignore_index=True)
# print(final_report)

# Find the model with the highest score for each metric
best_accuracy_model = final_report.loc[final_report['Accuracy Score'].idxmax()]['Model']
best_jaccard_model = final_report.loc[final_report['Jaccard Index'].idxmax()]['Model']
best_f1_model = final_report.loc[final_report['F1 Score'].idxmax()]['Model']
best_mae_model = final_report.loc[final_report['Mean Absolute Error'].idxmin()]['Model']
best_mse_model = final_report.loc[final_report['Mean Squared Error'].idxmin()]['Model']
best_r2_model = final_report.loc[final_report['R2 Score'].idxmax()]['Model']
best_log_loss_model = final_report.loc[final_report['Log Loss'].idxmin()]['Model']

# Create a dictionary with the best models for each metric
best_models = {
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score', 'Mean Absolute Error', 'Mean Squared Error', 'R2 Score', 'Log Loss'],
    'Best Model': [best_accuracy_model, best_jaccard_model, best_f1_model, best_mae_model, best_mse_model, best_r2_model, best_log_loss_model]
}

# Convert the dictionary into a DataFrame
best_models_df = pd.DataFrame(best_models)

# Print the DataFrame
print(best_models_df)

# print(f"Best model based on Accuracy Score: {best_accuracy_model}")
# print(f"Best model based on Jaccard Index: {best_jaccard_model}")
# print(f"Best model based on F1 Score: {best_f1_model}")
# print(f"Best model based on Log Loss: {best_log_loss_model}")

best_head = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(best_models_df.columns),
        fill_color='paleturquoise',
        align='left',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[best_models_df[col] for col in best_models_df.columns],
        fill_color='lavender',
        align='left',
        height=25,  # Adjust the height of the cells
        # line=dict(color='black', width=1),  # Add border to cells
        font=dict(size=12)  # Adjust font size
    )
)])

best_head.update_layout(
    margin=dict(l=270, r=0, t=30, b=30),  # Remove margins
    height=300,
    width=1800,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# ========================== Final Model Evaluation Table ========================== #

final_head = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(final_report.columns),
        fill_color='paleturquoise',
        align='left',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[final_report[col] for col in final_report.columns],
        fill_color='lavender',
        align='left',
        height=25,  # Adjust the height of the cells
        # line=dict(color='black', width=1),  # Add border to cells
        font=dict(size=12)  # Adjust font size
    )
)])

final_head.update_layout(
    margin=dict(l=270, r=0, t=30, b=30),  # Remove margins
    height=250,
    width=1800,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
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
    width=2800,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Rain Prediction in Australia', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%206%20-%20Rain%20Prediciton%20in%20Australia/australia_rain_data.py',
        className='btn')
    ]),

# Data Table 1
html.Div(
    className='row0',
    children=[
        html.Div(
            className='table',
            children=[
                html.H1(
                    className='table-title',
                    children='Australia Weather Data Table'
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

# Data Table 2
html.Div(
    className='row0',
    children=[
        html.Div(
            className='table',
            children=[
                html.H1(
                    className='table-title',
                    children='Algorithm Evaluation Table'
                )
            ]
        ),
        html.Div(
            className='table2', 
            children=[
                dcc.Graph(
                    className='data1',
                    figure=final_head
                )
            ]
        )
    ]
),

# Data Table 3
html.Div(
    className='row0',
    children=[
        html.Div(
            className='table',
            children=[
                html.H1(
                    className='table-title',
                    children='Best Performance Table'
                )
            ]
        ),
        html.Div(
            className='table2', 
            children=[
                dcc.Graph(
                    className='data1',
                    figure=best_head
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
                    # figure=
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

# updated_path = 'data/australia_weather_data.csv'
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