
# =============================== Imports ============================= #

from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import Image
import pydot
from PIL import Image
import pandas as pd
import os
import base64
from io import BytesIO
import dash
from dash import dcc, html
from dash.development.base_component import Component

# ========================== Load Data ==========================

# my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/drug200.csv'
file_path = os.path.join(script_dir, data_path)
my_data = pd.read_csv(file_path, delimiter=",") #delimiter is the separator in the csv file

# print(current_dir)
# print(script_dir)

# ========================== Data Exploration ========================== #

# Preview Data
# print(my_data.head())

# Summarize the data
# print("DF Shape:", my_data.shape)
# print("DTypes: \n", my_data.dtypes)
# print("Description: \n", my_data.describe())
# print("Info:", my_data.info())
# print("Columns:", my_data.columns)

# ========================== Pre Processing ========================== #

# X values are the features we are using to predict the target value
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# print(X[0:5])

# Below we are converting the categorical data into numerical data
# Sex 
le_sex = preprocessing.LabelEncoder() # LabelEncoder encodes labels with a value between 0 and n_classes-1
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) # Transform the data into numerical data
# print(le_sex)

# BP
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# Cholesterol
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

# print("Feature Variables: \n", X[0:5])

# Target variable
y = my_data["Drug"]
# print("Target variables: \n", y[0:5])

# ========================== Train / Test Split ========================== #

X_trainset, X_testset, y_trainset, y_testset = train_test_split(
  X, y, test_size=0.3, random_state=3)

# New shape of the training set
# print("X_trainset shape: ", X_trainset.shape)
# print("y_trainset shape: ", y_trainset.shape)

# New shape of the testing set
# print("X_testset shape: ", X_testset.shape)
# print("y_testset shape: ", y_testset.shape)

# ========================== Decision Tree ========================== #

drugTree = DecisionTreeClassifier(
  criterion="entropy", # measures the quality of a split
  max_depth = 4) # maximum depth of the tree
# print(drugTree) # it shows the default parameters

drugTree.fit(X_trainset,y_trainset)

# Prediction
predTree = drugTree.predict(X_testset)

# print ("Predictions: \n", predTree [0:5])
# print ("Actual values: \n", y_testset [0:5])

# Evaluation
dtree_acc = metrics.accuracy_score(y_testset, predTree)
# print("DecisionTrees's Accuracy: ", dtree_acc)

# ========================== Visualization ========================== #

#  Create Decision Tree classifer object
# To avoid a very large tree, we can set the max_depth to control the size of tree
dt = DecisionTreeClassifier(max_depth=4)

# Train Decision Tree Classifer with input and output
# dt = dt.fit(x_data,y_data)
dt.fit(X_trainset,y_trainset)

# Visualize Tree
fig, ax = plt.subplots(figsize=(20, 10))  # Set the width and height of the plot

# Set font size, feature name and max depth of the tree
features = my_data.columns.tolist()
tree.plot_tree(
  dt, # the data to be plotted
  ax=ax, # the axes to plot the data
  fontsize=10, 
  feature_names=features, 
  max_depth=4)

# Convert the Matplotlib figure to a base64-encoded string
buffer = BytesIO()
fig.savefig(buffer, format='png')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

# Save the plot to a file
fig_path = 'images/tree.png'
img_path = os.path.join(script_dir, fig_path)
fig.savefig(img_path)

# The function will return the accuracy. If it shows 0.9, it means 90% of the data can be accurately/correctly predicted on testing dataset.
# print('Prediction Accuracy of DT:', dt.score(X_testset, y_testset))

# export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

# Convert the DOT file to a PNG image
# tree_path = 'images/tree.png'
# (graph,) = pydot.graph_from_dot_file('tree.dot')
# img_path = os.path.join(script_dir, tree_path)
# # graph.write_png('tree.png')
# graph.write_png(img_path)

# # Convert the DOT file to a PNG image using the dot command
# os.system('dot -Tpng tree.dot -o tree.png')

# # Display the image in the notebook
# print(Image(filename='tree.png'))

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Decision Tree', 
        className='title'),

        html.A(
        'Repo',
        href='https://github.com/CxLos/IBM-Practice-Labs/blob/main/12.%20Machine%20Learning%20with%20Python/module%203%20-%20Classification/decision_tree.py',
        className='btn')
    ]),

# ROW 1
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                   html.Img(src=f'data:image/png;base64,{img_base64}')
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

# updated_path = 'data/drug200.csv'
# data_path = os.path.join(script_dir, updated_path)
# my_data.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ============================== Update Dash ================================ #

# pip install dash --upgrade
# pip install dash-core-components --upgrade
# pip install dash-html-components --upgrade
# pip install dash-renderer --upgrade

# ========================================================================== #