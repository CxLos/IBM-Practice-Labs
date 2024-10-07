
# =============================== Imports ============================= #

from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import os
import dash
from dash import dcc, html

# ============================= Load Data ============================= #

# url = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/heart_disease.csv'
file_path = os.path.join(script_dir, data_path)
# df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)

  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Connect to the SQLite database
conn = sqlite3.connect(heart_disease)
cursor = conn.cursor()
heart_disease.to_sql("heart_disease", conn, if_exists='replace', index=False, method="multi")

# print(heart_disease.head())

# metadata 
# print(heart_disease.metadata) 
  
# variable information 
# print(heart_disease.variables) 


# ========================== 1. ========================== #

query1 = pd.read_sql_query("""
SELECT 
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    AVG(age) AS avg_age
FROM 
    heart_disease_prediction_dataset;
"""
,conn)

# ========================== 2. ========================== #

query = pd.read_sql_query("""
SELECT 
    gender,
    COUNT(*) AS patient_count
FROM 
    heart_disease_prediction_dataset
GROUP BY 
    gender;
"""
,conn)

# ========================== 3. ========================== #

query = pd.read_sql_query("""
SELECT 
    gender,
    COUNT(*) AS patient_count
FROM 
    heart_disease_prediction_dataset
GROUP BY 
    gender;
"""
,conn)

# ========================== 4. ========================== #

query = pd.read_sql_query("""
SELECT 
    cp,
    COUNT(*) AS pain_frequency
FROM 
    heart_disease_prediction_dataset
GROUP BY 
    cp;
"""
,conn)

# ========================== 5. ========================== #

query = pd.read_sql_query("""
SELECT 
    CASE
        WHEN age BETWEEN 20 AND 30 THEN '20-30'
        WHEN age BETWEEN 31 AND 40 THEN '31-40'
        WHEN age BETWEEN 41 AND 50 THEN '41-50'
        WHEN age BETWEEN 51 AND 60 THEN '51-60'
        WHEN age BETWEEN 61 AND 70 THEN '61-70'
        ELSE 'Above 70'
    END AS age_group,
    SUM(CASE WHEN num = 1 THEN 1 ELSE 0 END) AS heart_disease_count,
    SUM(CASE WHEN num = 0 THEN 1 ELSE 0 END) AS no_heart_disease_count
FROM 
    heart_disease_prediction_dataset
GROUP BY 
    age_group
ORDER BY 
    age_group;
"""
,conn)

# ========================== 7. ========================== #

query = pd.read_sql_query("""
SELECT MIN(age) AS youngest_male_patient, MAX(age) AS oldest_male_patient
FROM your_dataset_name
WHERE gender = 1;
"""
,conn)

query = pd.read_sql_query("""
SELECT MIN(age) AS youngest_female_patient, MAX(age) AS oldest_female_patient
FROM your_dataset_name
WHERE gender = 0;
"""
,conn)

# ========================== 8. ========================== #

query = pd.read_sql_query("""
SELECT
    CASE
        WHEN age BETWEEN 20 AND 30 THEN '20-30'
        WHEN age BETWEEN 31 AND 40 THEN '31-40'
        WHEN age BETWEEN 41 AND 50 THEN '41-50'
        WHEN age BETWEEN 51 AND 60 THEN '51-60'
        WHEN age BETWEEN 61 AND 70 THEN '61-70'
        ELSE 'Above 70'
    END AS age_group,
    SUM(CASE WHEN num = 1 THEN 1 ELSE 0 END) AS heart_disease_count,
    SUM(CASE WHEN num = 0 THEN 1 ELSE 0 END) AS no_heart_disease_count
FROM your_dataset_name
GROUP BY age_group
ORDER BY age_group;
"""
,conn)

# ========================== 9. ========================== #

query = pd.read_sql_query("""
SELECT
    CASE
        WHEN age BETWEEN 20 AND 30 THEN '20-30'
        WHEN age BETWEEN 31 AND 40 THEN '31-40'
        WHEN age BETWEEN 41 AND 50 THEN '41-50'
        WHEN age BETWEEN 51 AND 60 THEN '51-60'
        WHEN age BETWEEN 61 AND 70 THEN '61-70'
        ELSE 'Above 70'
    END AS age_group,
    MAX(thalach) AS max_heart_rate
FROM your_dataset_name
GROUP BY age_group
ORDER BY age_group;
"""
,conn)
# ========================== 10. ========================== #

query = pd.read_sql_query("""
SELECT 
    (COUNT(CASE WHEN fbs = 1 THEN 1 END) * 100.0 / COUNT(*)) AS percentage_high_fbs
FROM your_dataset_name;
"""
,conn)
# ========================== 11. ========================== #

query = pd.read_sql_query("""
SELECT 
    SUM(CASE WHEN restecg > 0 THEN 1 ELSE 0 END) AS abnormal_results_count,
    SUM(CASE WHEN restecg = 0 THEN 1 ELSE 0 END) AS normal_results_count,
    (SUM(CASE WHEN restecg > 0 THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN restecg = 0 THEN 1 ELSE 0 END)) AS abnormal_to_normal_ratio
FROM your_dataset_name;
"""
,conn)
# ========================== 12. ========================== #

query = pd.read_sql_query("""
SELECT 
    COUNT(*) AS reversible_thalassemia_count
FROM your_dataset_name
WHERE thal = 7;
"""
,conn)
# ========================== 13. ========================== #

query = pd.read_sql_query("""
SELECT 
    AVG(age) AS average_age
FROM your_dataset_name
WHERE cp IS NOT NULL;
"""
,conn)
# ========================== 14. ========================== #

query = pd.read_sql_query("""
SELECT 
    ca AS major_vessels_colored,
    COUNT(*) AS patient_count
FROM your_dataset_name
GROUP BY ca
ORDER BY ca;
"""
,conn)

# ========================== DataFrame Table ========================== #

# df_table = go.Figure(data=[go.Table(
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

# df_table.update_layout(
#     margin=dict(l=50, r=50, t=30, b=40),  # Remove margins
#     height=400,
#     width=1500,  # Set a smaller width to make columns thinner
#     paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
#     plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
# )

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
                    # figure=df_table
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

# if __name__ == '__main__':
#     app.run_server(debug=
#                    True)
                #    False)

# ================================ Export Data =============================== #

updated_path = 'data/heart_disease.csv'
data_path = os.path.join(script_dir, updated_path)
heart_disease.to_csv(data_path, index=False)
print(f"DataFrame saved to {data_path}")

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