# IMPORTS --------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
import datetime as dt

# DATA ------------------------------------------------------------------------------------------------

data = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

df = pd.read_csv(data)

# --------------------------------------------- PART I ----------------------------------------------

# 1.1 Develop a *Line chart* using the functionality of pandas to show how automobile sales fluctuate from year to year

# df_line = df.groupby(df['Year'])['Automobile_Sales'].mean()

# plt.figure(figsize=(10, 6))
# df_line.plot(kind = 'line')
# plt.xlabel('Year')
# plt.ylabel('Sales')
# plt.title('Automobile Sales Over the Years')
# plt.xticks(list(range(1980,2024)), rotation = 75)
# plt.text(1982, 650, '1981-82 Recession')
# # plt.text(......, ..., '..............')
# plt.legend()
# plt.show()

# 1.2 Plot different lines for categories of vehicle type and analyse the trend to answer the question Is there a noticeable difference in sales trends between different vehicle types during recession periods?

# df_Mline = df.groupby(['Year','Vehicle_Type'], as_index=False)['Automobile_Sales'].sum()
# df_Mline.set_index('Year', inplace=True)
# df_Mline = df_Mline.groupby(['Vehicle_Type'])['Automobile_Sales']
# df_Mline.plot(kind='line')

# plt.xlabel('Year')
# plt.ylabel('Sales')
# plt.title('Sales Trend Vehicle-wise during Recession')
# plt.legend()
# plt.show()

# 1.3 Use the functionality of **Seaborn Library** to create a visualization to compare the sales trend per vehicle type for a recession period with a non-recession period.

# df1 = df.groupby('Recession')['Automobile_Sales'].mean().reset_index()

# plt.figure(figsize=(10,6))
# sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession',  data=df1)
# plt.xlabel('Recession Status')
# plt.ylabel('Sales')
# plt.title('Average Automobile Sales during Recession and Non-Recession')
# plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
# plt.show()

# Group by vehicle type

# plt.figure(figsize=(10, 6))
# sns.barplot(x='Recession', y='Automobile_Sales', hue='Vehicle_Type', data=dd)
# plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
# plt.xlabel('Recession Status')
# plt.ylabel('Sales')
# plt.title('Vehicle-Wise Sales during Recession and Non-Recession Period')

# plt.show()

# 1.4 Use sub plotting to compare the variations in GDP during recession and non-recession period by developing line plots for each period.

# rec_data = df[df['Recession'] == 1]
# non_rec_data = df[df['Recession'] == 0]

# #Figure
# fig=plt.figure(figsize=(12, 6))

# # Axes
# ax0 = fig.add_subplot(1, 2, 1) 
# ax1 = fig.add_subplot(1,2,2 )

# sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession', ax=ax0)
# ax0.set_xlabel('Year')
# ax0.set_ylabel('GDP')
# ax0.set_title('GDP Variation during Recession Period')

# sns.lineplot(x='Year', y='GDP', data=non_rec_data, label='Recession', ax=ax1)
# ax1.set_xlabel('Year')
# ax1.set_ylabel('GDP')
# ax1.set_title('GDP Variation during Non-Recession Period')

# plt.tight_layout()
# plt.show()

# 1.5 Develop a Bubble plot for displaying the impact of seasonality on Automobile Sales.

# non_rec_data = df[df['Recession'] == 0]

# size=non_rec_data['Seasonality_Weight'] 

# sns.scatterplot(data=non_rec_data, x='Month', y='Automobile_Sales', size=size, hue='Seasonality_Weight')

# plt.xlabel('Month')
# plt.ylabel('Automobile_Sales')
# plt.title('Seasonality impact on Automobile Sales')

# plt.show()

# 1.6 Use the functionality of Matplotlib to develop a scatter plot to identify the correlation between average vehicle price relate to the sales volume during recessions. From the data, develop a scatter plot to identify if there a correlation between consumer confidence and automobile sales during recession period? 

# rec_data = df[df['Recession'] == 1]
# non_rec_data = df[df['Recession'] == 0]

# plt.figure(figsize=(10, 6))

# plt.scatter(rec_data['Consumer_Confidence'], rec_data['Automobile_Sales'], label='Recession')
# plt.scatter(non_rec_data['Consumer_Confidence'], non_rec_data['Automobile_Sales'], label='Non Recession')

# plt.xlabel('Consumer Confidence')
# plt.ylabel('Automobile Sales')
# plt.title('Consumer Confidence and Automobile Sales during Recessions')
# plt.legend()
# plt.show()

# # Plot another scatter plot and title it as 'Relationship between Average Vehicle Price and Sales during Recessions'

# plt.figure(figsize=(10, 6))
# plt.scatter(non_rec_data['Price'], non_rec_data['Automobile_Sales'], color='blue', label='Non-Recession')
# plt.scatter(rec_data['Price'], rec_data['Automobile_Sales'], color='red', label='Recession')


# plt.xlabel('Average Vehicle Price')
# plt.ylabel('Automobile_Sales')
# plt.title('Relationship between Average Vehicle Price and Sales during Recessions')
# plt.legend()
# plt.show()

# 1.7 Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession and non-recession periods.

# rec_data = df[df['Recession'] == 1]
# non_rec_data = df[df['Recession'] == 0]

# RAtotal = rec_data['Advertising_Expenditure'].sum()
# NRAtotal = non_rec_data['Advertising_Expenditure'].sum()

# # Create a pie chart for the advertising expenditure 
# plt.figure(figsize=(8, 6))

# labels = ['Recession', 'Non-Recession']
# sizes = [RAtotal, NRAtotal]
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# plt.title('Advertising Expenditure of XYZAutomotive')

# plt.show()

# 1.8 Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession for each vehicle type.

# r_df = df[df['Recession'] == 1]

# # New Data
# exec_data = r_df[r_df['Vehicle_Type'] == 'Executivecar']
# med_data = r_df[r_df['Vehicle_Type'] == 'Mediumfamilycar']
# sml_data = r_df[r_df['Vehicle_Type'] == 'Smallfamilycar']
# sport_data = r_df[r_df['Vehicle_Type'] == 'Sports']
# mini_data = r_df[r_df['Vehicle_Type'] == 'Supperminicar']

# # Sizes
# exec_total = exec_data['Advertising_Expenditure'].sum()
# med_total = med_data['Advertising_Expenditure'].sum()
# sml_total = sml_data['Advertising_Expenditure'].sum()
# sport_total = sport_data['Advertising_Expenditure'].sum()
# mini_total = mini_data['Advertising_Expenditure'].sum()

# # pie Chart
# plt.figure(figsize=(8, 6))

# labels = ['Executivecar', 'Mediumfamilycar','Smallfamilycar','Sports','Supperminicar']
# sizes = [exec_total, med_total, sml_total, sport_total, mini_total]
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# plt.title('Advertising Expenditure of XYZAutomotive by Vehicle Type during')

# plt.show()

# 1.9 Develop a lineplot to analyse the effect of the unemployment rate on vehicle type and sales during the Recession Period.

# r_df = df[df['Recession'] == 1]

# sns.lineplot(data=r_df, x='unemployment_rate', y='Automobile_Sales',
#              hue='Vehicle_Type', style='Vehicle_Type', markers='o', err_style=None)
# plt.title('Automobile Sales During Recession')
# plt.ylim(0,850)
# plt.legend(loc=(0.05,.3))

# 1.10 Create a map on the hightest sales region/offices of the company during recession period

# geo = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/us-states.json'

# r_df = df[df['Recession'] == 1]
# sales_by_city = r_df.groupby('City')['Automobile_Sales'].sum().reset_index()
# print(sales_by_city)

# world_map = folium.Map(location=[0, 0], zoom_start=2)

# # generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
# folium.Choropleth(
#     geo_data=geo,
#     data=df,
#     columns=['City', 'Automobile_Sales'],
#     key_on='feature.properties.name',
#     fill_color='YlOrRd', 
#     fill_opacity=0.7, 
#     line_opacity=0.2,
#     legend_name='Sales by Region During Recession',
#     reset=True
# ).add_to(world_map)

# -------------------------------------------- PART II -----------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(children=[ 
    
    # Title
    html.H1('Automobile Sales Statistics Dashboard', 
        style={'textAlign': 'center', 'color': '#503D36',
        'font-size': 24}),

    # Dropdown Report
    html.Div([
        html.H2('Select Report:', style={'margin-right': '2em'}),
        dcc.Dropdown(options=[
            {'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'},
            {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'}],
            id='dropdown-statistics', placeholder='Select Report Type'),

        # Dropdown Year
        html.Div([
            html.H2('Select Year:', style={'margin-right': '2em'}),
            dcc.Dropdown(df.Year.unique(), value=2013, id='select-year')]),

          # html.Div([
          # html.H2('Select Year:', style={'margin-right': '2em'}),
          # dcc.Dropdown(options=[{'label': year, 'value': year} for year in df['Year'].unique()],
          #               value=2013, id='select-year')]),
    ]),

    html.Br(),
    html.Br(), 

    # Segment 1
    # html.Div([
    #     html.Div(dcc.Graph(id='chart-grid', className='chart-grid')),
    #     html.Div(dcc.Graph(id='R_chart2', className='chart-grid'))],
    #     style={'display': 'flex'}),

    # # Segment 2
    # html.Div([
    #     html.Div(dcc.Graph(id='R_chart3', className='chart-grid')),
    #     html.Div(dcc.Graph(id='R_chart4', className='chart-grid'))],
    #     style={'display': 'flex'}),

    # html.Div([
    #     html.Div(
    #         id='output-container', 
    #         className='chart-grid', 
    #         style={'display': 'flex'}),
    #     ]),

    html.Div([
        html.Div(id='output-container', 
                 className='chart-grid', 
                  style={'display': 'flex', 'text-align':'center'})
        ])

     # Segment 1
          # html.Div([
          #           html.Div(dcc.Graph(id='R_chart1')),
          #           html.Div(dcc.Graph(id='R_chart2'))],
          #             style={'display': 'flex'}),

          # # Segment 2
          #   html.Div([
          #           html.Div(dcc.Graph(id='R_chart3')),
          #           html.Div(dcc.Graph(id='R_chart4'))],
          #             style={'display': 'flex'}),
])

# Callback to enable input container
@app.callback(
    Output(component_id='select-year', component_property='disabled'),
    Input(component_id='dropdown-statistics', component_property='value'))

def update_input_container(input_rec):
    if input_rec == 'Yearly Statistics': 
        return False
    else: 
        return True

# Callback for plotting
@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='dropdown-statistics', component_property='value'), 
     Input(component_id='select-year', component_property='value')])
  

def update_output_container(input_rec, input_year):
    if input_rec == 'Recession Period Statistics':
        print(input_rec)

        data = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

        df = pd.read_csv(data)
        
        # Filter the data for recession periods
        recession_data = df[df['Recession'] == 1]

        #Plot 1 Automobile sales fluctuate over Recession Period (year wise) using line chart
        yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(
            figure=px.line(yearly_rec, 
                x='Year',
                y='Automobile_Sales',
                title="Automobile Sales During Recession"))

        #Plot 2 Calculate the average number of vehicles sold by vehicle type and represent as a Bar chart
        rec_bar = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        R_chart2 = dcc.Graph(
            figure=px.bar(rec_bar, x='Vehicle_Type', y='Automobile_Sales', title='Average Number of Vehicles Sold by Vehicle Type'))

        # Plot 3 : Pie chart for total expenditure share by vehicle type during recessions
        exp_rec = recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        R_chart3 = dcc.Graph(
            figure=px.pie(exp_rec, values='Advertising_Expenditure', names='Vehicle_Type', title="Advertising Expenditure by Vehicle Type"))

        # Plot 4 Develop a Bar chart for the effect of unemployment rate on vehicle type and sales
        un_data = recession_data.groupby('Vehicle_Type').agg({'Automobile_Sales': 'mean', 'unemployment_rate': 'mean'}).reset_index()
        R_chart4 = dcc.Graph(
            figure=px.bar(un_data, x='Vehicle_Type', y='Automobile_Sales', color='unemployment_rate', 
                        title='Effect of Unemployment Rate on Vehicle Type and Sales',
                        labels={'Automobile_Sales': 'Average Sales', 'unemployment_rate': 'unemployment rate'},
                        # color_continuous_scale=px.colors.sequential.Viridis
                        ))

        # return R_chart1, R_chart2, R_chart3

        # return [
        #     html.Div(className='chart-item', children=[
        #         html.Div(children=[R_chart1]),
        #         html.Div(children=[R_chart2])
        #     ]),
        #     html.Div(className='chart-item', children=[
        #         html.Div(children=[R_chart3]),
        #         html.Div(children=[R_chart4])
        #     ])
        # ]

        return html.Div(id='output-container', children=[
                  html.Div(className='chart-item', children=[
                  html.Div(children=[R_chart1]),
                  html.Div(children=[R_chart2]),
                ], style={'display':'flex'}),

                  html.Div(className='chart-item', children=[
                  html.Div(children=[R_chart3]),
                  html.Div(children=[R_chart4])
                ], style={'display':'flex'})
              ])

    # Yearly Statistic Report Plots                             
    elif (input_year and input_rec == 'Yearly Statistics'):

        data = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

        df = pd.read_csv(data)
          
        print(input_year)
        
        yearly_data = df[df['Year'] == input_year]

        # Plot 1 :Yearly Automobile sales using line chart for the whole period.
        yas = df.groupby('Year')['Automobile_Sales'].sum().reset_index()
        Y_chart1 = dcc.Graph(
            figure=px.line(
                yas,
                x='Year',
                y='Automobile_Sales',
                title='Automobile Sales by Year'
            ))

        # Plot 2 :Total Monthly Automobile sales using line chart.

        mas = yearly_data.groupby('Month')['Automobile_Sales'].sum().reset_index()
        Y_chart2 = dcc.Graph(
            figure = px.line(
                mas,
                x='Month',
                y='Automobile_Sales',
                title='Automobile Sales by Month'))

        # Plot 3 bar chart for average number of vehicles sold per type during the given year
        avr_vdata = yearly_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()

        Y_chart3 = dcc.Graph(
            figure = px.bar (
                avr_vdata,
                x='Vehicle_Type',
                y='Automobile_Sales',
                title='Average Vehicles Sold by Vehicle Type in the year {}'.format(input_year)))

        # Plot 4 Total Advertisement Expenditure for each vehicle using pie chart
        hg = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()

        # exec_data = recession_data[recession_data['Vehicle_Type'] == 'Executivecar']
        # med_data = recession_data[recession_data['Vehicle_Type'] == 'Mediumfamilycar']
        # sml_data = recession_data[recession_data['Vehicle_Type'] == 'Smallfamilycar']
        # sport_data = recession_data[recession_data['Vehicle_Type'] == 'Sports']
        # mini_data = recession_data[recession_data['Vehicle_Type'] == 'Supperminicar']

        # Calculate total advertising expenditure for each vehicle type
        # exec_total = exec_data['Advertising_Expenditure'].sum()
        # med_total = med_data['Advertising_Expenditure'].sum()
        # sml_total = sml_data['Advertising_Expenditure'].sum()
        # sport_total = sport_data['Advertising_Expenditure'].sum()
        # mini_total = mini_data['Advertising_Expenditure'].sum()

        # Create a DataFrame
        # data = {
        #     'Vehicle_Type': ['Executivecar', 'Mediumfamilycar', 'Smallfamilycar', 'Sports', 'Supperminicar'],
        #     'Advertising_Expenditure': [exec_total, med_total, sml_total, sport_total, mini_total]
        # }
        # df7 = pd.DataFrame(data)

        # Generate pie chart
        Y_chart4 = dcc.Graph(
            figure= px.pie(
                data_frame=hg, 
                values='Advertising_Expenditure', 
                names='Vehicle_Type', 
                title='Advertising Expenditure of XYZAutomotive by Vehicle Type')
        )

        # return [
        #     html.Div(className='chart-item', children=[
        #         html.Div(children=Y_chart1),
        #         html.Div(children=Y_chart2)],
        #         style={'display': 'flex'}),

        #     html.Div(className='chart-item', children=[
        #         html.Div(children=Y_chart3),
        #         html.Div(children=Y_chart4)],
        #         style={'display': 'flex'})
        # ]
        return html.Div(id='output-container', children=[
            
                  html.Div(className='chart-item', children=[
                    html.Div(children=[Y_chart1]),
                    html.Div(children=[Y_chart2])],
                      style={'display': 'flex', 'text-align':'center'}),

                  html.Div(className='chart-item', children=[
                    html.Div(children=[Y_chart3]),
                    html.Div(children=[Y_chart4])],
                      style={'display': 'flex'})
        ])
    
      # return html.Div(id='output-container', children=[
      #             html.Div(className='chart-item', children=[
      #             html.Div(children=[R_chart1]),
      #             html.Div(children=[R_chart2]),
      #           ], style={'display':'flex'}),

      #             html.Div(className='chart-item', children=[
      #             html.Div(children=[R_chart3]),
      #             html.Div(children=[R_chart4])
      #           ], style={'display':'flex'})
      #         ])


if __name__ == '__main__':
    app.run_server()

# PRINTS ----------------------------------------------------------------------------------------------

# print(df[df['Recession'] == 1])
# print(df.shape)
# print(df.describe())
# print(df.columns)
# print(df1.head())
# print(df10.head())
# print(df['City'].unique())
# print()

# SAVE ----------------------------------------------------------------------------------------------

# df.to_csv('historical_automobile_sales.csv')
# df.to_json('us-states.json')
# fig.write_html('bicycle_sales.html')
# --------------------------------------------------------------------------------------------------