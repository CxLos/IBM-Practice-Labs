
# ================================ Imports ======================== #

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import base64
import os
from io import BytesIO
import dash
from dash import dcc, html
from dash.development.base_component import Component

# ================================ Data =========================== #

# churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/churn_data.csv'
file_path = os.path.join(script_dir, data_path)
churn_df = pd.read_csv(file_path, delimiter=",")

# ========================== Data Exploration ========================== #

# Preview Data
print(my_data.head())

# Summarize the data
# print("DF Shape:", my_data.shape)
# print("DTypes: \n", my_data.dtypes)
# print("Description: \n", my_data.describe())
# print("Info:", my_data.info())
# print("Columns:", my_data.columns)

# ========================== Pre Processing ========================== #

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

# ========================== Train / Test Split ========================== #



# ========================== Decision Tree ========================== #



# ========================== Visualization ========================== #



# ========================== Confusion Matrix ========================== #

# Generate the confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion matrix')

# Convert the Matplotlib figure to a base64-encoded string
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Logistic Regression', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%203%20-%20Classification/regression_tree.py',
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
                  

                )
            ]
        ),
        html.Div(
            className='graph1',
            children=[
                html.H1("Confusion Matrix"),
                html.Img(src=f'data:image/png;base64,{img_base64}')
            ]
        ),
    ]
),
])

# if __name__ == '__main__':
#     app.run_server(debug=
#                    True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/churn_data.csv'
# data_path = os.path.join(script_dir, updated_path)
# churn_df.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #