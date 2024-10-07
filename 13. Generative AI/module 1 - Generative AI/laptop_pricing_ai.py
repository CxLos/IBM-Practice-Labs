
# =============================== Imports ============================= #

from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pandas as pd
import os
import dash
from dash import dcc, html

# ============================= Load Data ============================= #

# url = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/laptop_pricing_data.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

# ========================== 1. ========================== #

# Assuming the first rows of the file are the headers, you don't need to specify any additional parameters

# Additional details:
# - The `pd.read_csv()` function is used to read a CSV file into a Pandas data frame.
# - By default, it assumes that the first row of the file contains the headers for the data.
# - If your file doesn't have headers, you can specify `header=None` as an additional parameter.
# - You can also specify other parameters, such as `sep` to specify the delimiter used in the file.
# - Make sure you have the Pandas library installed in your Python environment before running this code.

# ========================== 2. ========================== #

# Assuming you already have a Pandas data frame named 'df'
# Identify columns with missing values
columns_with_missing_values = df.columns[df.isnull().any()]
# Additional details:
# - The `df.isnull()` function returns a Boolean data frame where each cell is True if it contains a missing value (NaN), and False otherwise.
# - The `df.columns` attribute returns the column labels of the data frame.
# - The `.any()` method returns a Boolean Series indicating whether any value in the given axis (in this case, columns) is True.
# - Finally, the `.columns` attribute is used to retrieve the column labels where the condition is True.
# You can now use the 'columns_with_missing_values' variable to further analyze or handle the columns with missing values.

# ========================== 3. ========================== #

# Assuming you already have a Pandas data frame named 'df'
# Replace missing values in the 'Screen_Size_cm' column with the most frequent value
most_frequent_value = df['Screen_Size_cm'].mode()[0]
df['Screen_Size_cm'].fillna(most_frequent_value, inplace=True)
# Replace missing values in the 'Weight_kg' column with the mean value
mean_value = df['Weight_kg'].mean()
df['Weight_kg'].fillna(mean_value, inplace=True)
# Additional details:
# - The `.mode()` method is used to calculate the most frequent value in a column.
# - The `[0]` indexing is used to retrieve the most frequent value from the resulting Series.
# - The `.fillna()` method is used to replace missing values with a specified value.
# - The `inplace=True` parameter is used to modify the original data frame instead of creating a new one.
# You can now use the modified 'df' data frame, which has the missing values replaced according to the guidelines.

# ========================== 4. ========================== #

# Assuming you already have a Pandas data frame named 'df'
# Change the data type of 'Screen_Size_cm' and 'Weight_kg' to float
df['Screen_Size_cm'] = df['Screen_Size_cm'].astype(float)
df['Weight_kg'] = df['Weight_kg'].astype(float)
# Additional details:
# - The `.astype()` method is used to change the data type of a column.
# - In this case, we're specifying `float` as the desired data type.
# - Make sure the columns contain numeric values that can be converted to float.
# - If there are any non-numeric values in the columns, the conversion will raise an error.
# You can now use the modified 'df' data frame, which has the data types of 'Screen_Size_cm' and 'Weight_kg' changed to float.

# ========================== 5. ========================== #

# Assuming you already have a Pandas data frame named 'df'

# Convert 'Screen_Size_cm' from centimeters to inches and modify the attribute name
df['Screen_Size_inch'] = df['Screen_Size_cm'] * 0.393701
df.drop('Screen_Size_cm', axis=1, inplace=True)

# Convert 'Weight_kg' from kilograms to pounds and modify the attribute name
df['Weight_pounds'] = df['Weight_kg'] * 2.20462
df.drop('Weight_kg', axis=1, inplace=True)

# Additional details:
# - The code multiplies the values under 'Screen_Size_cm' by 0.393701 to convert centimeters to inches.
# - The resulting values are stored in a new attribute named 'Screen_Size_inch'.
# - The original 'Screen_Size_cm' attribute is dropped from the data frame using the `.drop()` method.
# - Similarly, the code multiplies the values under 'Weight_kg' by 2.20462 to convert kilograms to pounds.
# - The resulting values are stored in a new attribute named 'Weight_pounds'.
# - The original 'Weight_kg' attribute is dropped from the data frame.

# You can now use the modified 'df' data frame, which has the contents and attribute names modified as required.

# ========================== 6. ========================== #

# Assuming you already have a Pandas data frame named 'df'

# Normalize the content under 'CPU_frequency' with respect to its maximum value
max_value = df['CPU_frequency'].max()
df['CPU_frequency'] = df['CPU_frequency'] / max_value

# Additional details:
# - The code calculates the maximum value of the 'CPU_frequency' attribute using the `.max()` method.
# - It then divides the values under 'CPU_frequency' by the maximum value to normalize them.
# - The resulting normalized values overwrite the original values in the 'CPU_frequency' attribute.

# You can now use the modified 'df' data frame, which has the content under the 'CPU_frequency' attribute normalized.

# ========================== 7. ========================== #

# Assuming you already have a Pandas data frame named 'df'

# Convert the 'Screen' attribute into indicator variables
df1 = pd.get_dummies(df['Screen'], prefix='Screen')

# Append df1 into the original data frame df
df = pd.concat([df, df1], axis=1)

# Drop the original 'Screen' attribute from the data frame
df.drop('Screen', axis=1, inplace=True)

# Additional details:
# - The `pd.get_dummies()` function is used to convert a categorical attribute into indicator variables.
# - The resulting indicator variables are stored in a new data frame named 'df1'.
# - The `prefix` parameter is used to specify the naming convention for the indicator variables.
# - The `pd.concat()` function is used to concatenate the original data frame 'df' and 'df1' along the column axis (axis=1).
# - The resulting concatenated data frame is assigned back to 'df'.
# - Finally, the `.drop()` method is used to drop the original 'Screen' attribute from 'df'.

# You can now use the modified 'df' data frame, which has the 'Screen' attribute converted into indicator variables, appended, and the original attribute dropped.

# ========================== 8. ========================== #

df1['Price Euros'] = df1['Price'] * 0.90

# ========================== 9. ========================== #

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Perform min-max normalization on the 'CPU_frequency' parameter
#  this will scale the values between 0 and 1
df['CPU_frequency'] = scaler.fit_transform(df[['CPU_frequency']])

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
    width=1500,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# =============================== Dash App =============================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(children=[ 

    html.Div(className='divv', children=[ 
        
        html.H1('Laptop Pricing Prediction', 
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
                    children='Laptop Pricing Data Table'
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

# updated_path = 'data/laptop_pricing_data.csv'
# data_path = os.path.join(script_dir, updated_path)
# url.to_csv(data_path, index=False)
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