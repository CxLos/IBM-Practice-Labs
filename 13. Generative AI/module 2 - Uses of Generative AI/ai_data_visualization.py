# ========================= Imports ======================== #

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_ibm import WatsonxLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pydantic import BaseModel
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import pipeline
import openai
import pandas as pd
import numpy as np
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

# ========================= Load Data ======================== #

# df = pd.read_csv(
#     "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
# )

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/student_mat.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

# print(current_dir)
# print(script_dir)
# print(df.info())

# ========================= Load LLM ======================== #

# Create a dictionary to store credential information
credentials = {
    # "url"    : "https://us-south.ml.cloud.ibm.com"
    "url"    : "https://api.au-syd.assistant.watson.cloud.ibm.com/instances/22a6647d-8b8e-40d7-886f-a2602cc332de"
    ,"apikey": "8UTbOq2-uDj0LfJWsPQ9dFiznxYyPG-pixhwy6lawX9r" 
}

# Indicate the model we would like to initialize. In this case, Llama 3 70B.
model_id    = 'meta-llama/llama-3-70b-instruct'

# Initialize some watsonx.ai model parameters
params = {
        GenParams.MAX_NEW_TOKENS: 256, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }
project_id  = "skills-network" # <--- note: specify "skills-network" as your project_id
space_id    = None
verify      = False

# Launch a watsonx.ai model
model = Model(
    model_id=model_id, 
    credentials=credentials, 
    params=params, 
    project_id=project_id, 
    space_id=space_id, 
    verify=verify
)

# Integrate the watsonx.ai model with the langchain framework
llm = WatsonxLLM(model = model)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    return_intermediate_steps=True  # set return_intermediate_steps=True so that model could return code that it comes up with to generate the chart
)

response = agent.invoke("how many rows of data are in this file?")
print(response)

# ========================= Face Huggers Transformers ======================== #

# Load a Hugging Face text generation pipeline (e.g., GPT-2)
# llm = pipeline("text-generation", model="gpt2")

# # Define a function to ask the model how many rows are in the DataFrame
# def ask_llm_about_dataframe(prompt):
    
#     # Include the actual data row count in the prompt
#     row_count = df.shape[0]
#     full_prompt = f"{prompt}. The dataset has {row_count} rows."
    
#     # Get the model's response
#     response = llm(full_prompt, max_length=50)
#     return response[0]['generated_text']

# # Test the LLM interaction
# question = "How many rows of data are in this file?"
# response = ask_llm_about_dataframe(question)
# # print(response)  # Output should include the row count response
# # print(df.shape[0])

# # Simulated response from a Transformer model
# response = {
#     'intermediate_steps': [
#         [
#             {
#                 'tool_input': 'Step 1; Step 2; Step 3'
#             }
#         ]
#     ]
# }

# # Process the tool input to replace '; ' with newline characters
# formatted_output = response['intermediate_steps'][0][0]['tool_input'].replace('; ', '\n')

# # Print the formatted output
# print(formatted_output)

# ========================= OpenAI ======================== #

# openai.api_key = "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"

# response = openai.Completion.create(
#     model="gpt-3.5-turbo",
#     prompt="How many rows of data are in this file?",
#     max_tokens=50
# )

# # response = openai.ChatCompletion.create(
# #     model="gpt-3.5-turbo",  # Specify the model you want to use
# #     messages=[
# #         {"role": "user", "content": "How many rows of data are in this file?"}
# #     ]
# # )

# # print(response.choices[0].text.strip())
# print(response['choices'][0]['message']['content'])


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
        
        html.H1('Student Alcohol Consumption Data', 
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

# updated_path = 'data/student_mat.csv'
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