
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

# length of test data
# print(X_test.shape, Y_test.shape)
# print(len(X_test), len(Y_test))

# 4. Create a logistic regression object  then create a  GridSearchCV object  <code>logreg_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.

LR = LogisticRegression()
LR.fit(X_train, Y_train)

parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}

GridSearchCV(
    estimator=LR,
    param_grid={'C': np.logspace(-5, 8, 15), 'penalty': ['l1', 'l2']},
    cv=10
)

# Perform the grid search with cross-validation
logreg_cv = GridSearchCV(
    LR, # The logistic regression model
    parameters, # The dictionary of parameters
    cv=5) # The number of folds

# Fit the model to the data (assuming X_train and y_train are your training data)
logreg_cv.fit(X_train, Y_train)

# Display the best parameters using the data attribute best_params and accuracy on the validation data using the data attribute best_score_
# print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
# print("accuracy :",logreg_cv.best_score_)

# 5. Calculate the accuracy on the test data using the method <code>score</code>:
# Calculate the accuracy on the test data using the method score:
# print("accuracy :", logreg_cv.score(X_test, Y_test))

# Confusion Matrix
yhat=logreg_cv.predict(X_test)
# plot_confusion_matrix(Y_test,yhat)

# 6. Create a support vector machine object then create a  GridSearchCV object  <code>svm_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
svm = SVC()

parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}

svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)

# print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
# print("accuracy :",svm_cv.best_score_)

# 7. Calculate the accuracy on the test data using the method <code>score</code>:
# print("accuracy :", svm_cv.score(X_test, Y_test))

# Confusion Matrix
yhat=svm_cv.predict(X_test)
# plot_confusion_matrix(Y_test,yhat)

# 8. Create a decision tree classifier object then  create a  GridSearchCV object  <code>tree_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
tree = DecisionTreeClassifier()

parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)

# print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
# print("accuracy :",tree_cv.best_score_)

# 9. Calculate the accuracy of tree_cv on the test data using the method <code>score</code>:
# print("accuracy :", tree_cv.score(X_test, Y_test))

# Confusion Matrix
yhat = tree_cv.predict(X_test)
# plot_confusion_matrix(Y_test,yhat)

# 10. Create a k nearest neighbors object then  create a  GridSearchCV object  <code>knn_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
              'p': [1,2]} # Create a k-NN classifier with 7 neighbors

KNN = KNeighborsClassifier()

knn_cv = GridSearchCV(KNN, parameters, cv=10)
knn_cv.fit(X_train, Y_train)

# print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
# print("accuracy :",knn_cv.best_score_)

# 11. Calculate the accuracy of knn_cv on the test data using the method <code>score</code>:
# print("accuracy :", knn_cv.score(X_test, Y_test))

# Confusion Matrix
yhat = knn_cv.predict(X_test)
# plot_confusion_matrix(Y_test,yhat)

# 12. Find the method that performs the best:
best_score = max(logreg_cv.best_score_, svm_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_)

# if best_score == logreg_cv.best_score_:
#     print('Logistic Regression')
# elif best_score == svm_cv.best_score_:
#     print('Support Vector Machine')
# elif best_score == tree_cv.best_score_:
#     print('Decision Tree')
# else:
#     print('K Nearest Neighbors')

# Decision Tree is the best method

# Scores of each model:

logreg_score = logreg_cv.best_score_
svm_score = svm_cv.best_score_
tree_score = tree_cv.best_score_
knn_score = knn_cv.best_score_


# Create a DataFrame with the model names and their corresponding accuracy scores
accuracy_scores = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest Neighbors'],
    'Accuracy Score': [logreg_score, svm_score, tree_score, knn_score]
})

# Create a bar chart using Plotly
fig = px.bar(
    accuracy_scores,
    x='Accuracy Score',
    y='Model',
    color='Model',
    title='Accuracy Scores of Different Models',
    text='Accuracy Score',
    # orientation='h',
    labels={'Accuracy Score': 'Accuracy Score'}
)

# Customize the layout
fig.update_layout(
    title_x=0.5,
    font=dict(family='Calibri', size=17, color='black'),
    yaxis=dict(title='Accuracy Score'),
    xaxis=dict(title='Model'),
    showlegend=False
)

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
                  figure=fig
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

if __name__ == '__main__':
    app.run_server(debug=
                   True)
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