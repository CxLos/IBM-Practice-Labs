# IMPORTS ---------------------------------------------------------------------------------------------------
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
# from dash import html
from dash.dependencies import Input, Output

# DATA--------------------------------------------------------------------------------------------------

airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})

# data = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\8. Data Visualization with Python\W4 - Creating Dashboards with Plotly and Dash\Data\airline_data.csv'

# airline_data = pd.read_csv(data,
                  #  encoding="ISO-8859-1",
                  #  dtype={'Div1Airport': str, 'Div1TailNum': str, 
                  #         'Div2Airport': str, 'Div2TailNum': str})

# CREATE DASH APPLICATION ------------------------------------------------------------------------------

# Create a dash application layout
app = dash.Dash(__name__)

# Get the layout of the application and adjust it.
# Create an outer division using html.Div and add title to the dashboard using html.H1 component
# Add a html.Div and core input text component
# Finally, add graph component.
app.layout = html.Div(
                children=[html.H1('Airline Performance Dashboard', 
                    # css properties
                    style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

                html.Div(["Input Year: ", dcc.Input(id='input-year', value='2010', 
                    type='number', style={'height':'50px', 'font-size': 35}),
                    html.Button('Submit', id='submit-button', n_clicks=0,
                        style={'height': '50px', 'font-size': 20})
                        ],
                style={'font-size': 40}),
                html.Br(),
                html.Br(),
                html.Div(dcc.Graph(id='line-plot')),
              ])

# add callback decorator
@app.callback( Output(component_id='line-plot', component_property='figure'),
               Input(component_id='input-year', component_property='value'))

# Add computation to callback function and return graph
def get_graph(entered_year):
    # Select 2019 data
    df =  airline_data[airline_data['Year']==int(entered_year)]
    
    # Group the data by Month and compute average over arrival delay time.
    line_data = df.groupby('Month')['ArrDelay'].mean().reset_index()
    fig = go.Figure(data=go.Scatter(x=line_data['Month'], y=line_data['ArrDelay'], mode='lines', marker=dict(color='green')))
    fig.update_layout(title='Month vs Average Flight Delay Time', xaxis_title='Month', yaxis_title='ArrDelay')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()

# Prints ------------------------------------------------------------------------------------------------

# print(airline_data.head())