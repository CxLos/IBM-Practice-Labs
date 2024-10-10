# ========================= Imports ======================== #

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
# -------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# -------------------------
import os
import dash
from dash import dcc, html

# ========================= Load Data ======================== #

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0271EN-SkillsNetwork/labs/v1/m3/data/used_car_price_analysis.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/used_car_price_analysis.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.values)
# print(df.corr())

# print(current_dir)
# print(script_dir)

# Value counts for each column
# for column in df.columns:
#     print(f"Value counts for column: {column} \n")
#     print(df[column].value_counts())
#     print("\n")

# ========================= 2. Data Preparation ======================== #

# Task 1: Identify columns with missing values and fill them with the average value of the columns
# df.fillna(df.mean(), inplace=True)
# df.dropna()

# missing values
# print(df.isnull().sum())

# Replace NaN values in 'tax' column with the mean
df['tax'].fillna(df['tax'].mean(), inplace=True)

# Task 2: Identify and drop duplicate entries
# df.drop_duplicates(inplace=True)

# print("Data cleaning complete. Missing values filled and duplicates dropped.")

# ====================== 3. Data Insights & Visualizations ===================== #

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['fuelType', 'transmission', 'model'])
# print(df_encoded.head())
# print(df_encoded.columns)


# only numerical columns:
df1 = df.select_dtypes(include=[np.number])

# 1. The 5 attributes with highest correlation to price
# Calculate the correlation matrix
correlation_matrix = df_encoded.corr()
# print(correlation_matrix)
# print(df1.corr())
# print(df_encoded.corr())

# Assuming 'Price' is the target attribute
target_attribute = 'price'

# Get the correlation values for the target attribute, excluding the target itself
correlations = correlation_matrix[target_attribute].drop(target_attribute)

# Identify the top 5 attributes with the highest correlation with the target attribute
top_5_attributes = correlations.abs().nlargest(5).index

print("Top 5 attributes with the highest correlation with the target attribute:", top_5_attributes)

# 2. Count the number of cars under each unique value of fuelType attribute.

# Specify the attribute/column you want to analyze
attribute = 'fuelType'  # Replace 'FuelType' with your specific attribute

# Count the number of entries for each unique value of the specified attribute
value_counts = df[attribute].value_counts()

# print(f"Number of entries for each unique value in the '{attribute}' column:")
# print(value_counts)

# 3. Create a Box plot to determine whether cars with automatic, manual or semi-auto type of transmission have more price outliers. Use the Seaborn library for creating the plot.

# Specify the source and target attributes
source_attribute = 'fuelType'  # Replace with your source attribute
target_attribute = 'price'    # Replace with your target attribute

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=source_attribute, y=target_attribute, data=df)
plt.title(f'Box Plot of {target_attribute} by {source_attribute}')
plt.xlabel(source_attribute)
plt.ylabel(target_attribute)
# plt.show()

# 4. Generate the regression plot between mpg parameter and the price to determine the correlation type between the two.

# Specify the source and target attributes
source_attribute = 'mileage'  # Replace with your source attribute
target_attribute = 'price'  # Replace with your target attribute

# Create the regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=source_attribute, y=target_attribute, data=df)
plt.title(f'Regression Plot of {target_attribute} vs {source_attribute}')
plt.xlabel(source_attribute)
plt.ylabel(target_attribute)
# plt.show()

# ====================== 4. Model Development & Evaluation ====================== #

# 1. Fit a linear regression model to predict the price using the feature mpg. Then calculate the R^2 and MSE values for the model.

# Specify the source and target attributes
source_attribute = 'mileage'  # Replace with your source attribute
target_attribute = 'price'    # Replace with your target attribute

# Prepare the data
X = df[[source_attribute]]
y = df[target_attribute]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print(f'R-squared: {r2}')
# print(f'Mean Squared Error: {mse}')

# 2. Fit a linear regression model to predict the price using the following set of features:

# year
# mileage
# tax
# mpg
# engineSize.
# Calculate the R^2 and MSE values for this model.

# Specify the source and target attributes
source_attributes = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
target_attribute = 'price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print(f'R-squared: {r2}')
# print(f'Mean Squared Error: {mse}')

# 3. For the same set of features as in the question above, create a pipeline model object that uses standard scalar, second degree polynomial features and a linear regression model. Calculate the R^2 value and the MSE value for this model.

# Specify the source and target attributes
source_attributes = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
target_attribute = 'price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures (degree 2), and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print(f'R-squared: {r2}')
# print(f'Mean Squared Error: {mse}')

# 4. For the same set of features, split the data into training and testing data parts. Assume testing part to be 20%. Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.

# Specify the source and target attributes
source_attributes = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
target_attribute = 'price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Ridge regression model with regularization parameter alpha=0.1
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print(f'R-squared: {r2}')
# print(f'Mean Squared Error: {mse}')

# 5. Perform a second order polynomial transform on both the training data and testing data created for the question above. Create and fit a Ridge regression object using the modified training data, set the regularisation parameter to 0.1, and calculate the R^2 and MSE utilising the modified test data.

# Specify the source and target attributes
source_attributes = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
target_attribute = 'price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures (degree 2), and Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=0.1))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print(f'R-squared: {r2}')
# print(f'Mean Squared Error: {mse}')

# 6. In the question above, perform a Grid Search on ridge regression for a set of values of alpha {0.01, 0.1, 1, 10, 100} with 4-fold cross validation to find the optimum value of alpha to be used for the prediction model.

# Specify the source and target attributes
source_attributes = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
target_attribute = 'price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures (degree 2), and Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge())
])

# Define the parameter grid for alpha
param_grid = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}

# Create and fit the GridSearchCV with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print(f'Best alpha: {grid_search.best_params_["ridge__alpha"]}')
# print(f'R-squared: {r2}')
# print(f'Mean Squared Error: {mse}')

# ========================== DataFrame Table ========================== #

# fig_head = go.Figure(data=[go.Table(
#     # columnwidth=[50, 50, 50],  # Adjust the width of the columns
#     header=dict(
#         values=list(df.columns),
#         fill_color='paleturquoise',
#         align='left',
#         height=30,  # Adjust the height of the header cells
#         # line=dict(color='black', width=1),  # Add border to header cells
#         font=dict(size=12)  # Adjust font size
#     ),
#     cells=dict(
#         values=[df[col] for col in df.columns],
#         fill_color='lavender',
#         align='left',
#         height=25,  # Adjust the height of the cells
#         # line=dict(color='black', width=1),  # Add border to cells
#         font=dict(size=12)  # Adjust font size
#     )
# )])

# fig_head.update_layout(
#     margin=dict(l=50, r=50, t=30, b=40),  # Remove margins
#     height=400,
#     width=2800,  # Set a smaller width to make columns thinner
#     paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
#     plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
# )

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Used Car Price Analysis', 
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
                    # figure=fig_head
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
                    # figure=final_head
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
                    # figure=best_head
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

# if __name__ == '__main__':
#     app.run_server(debug=
#                    True)
                #    False)

# ================================ Export Data =============================== #

# updated_path = 'data/used_car_price_analysis.csv'
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