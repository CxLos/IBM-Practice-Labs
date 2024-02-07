# IMPORTS ---------------------------------------------------------------------------------------------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# SCATTER PLOT -----------------------------------------------------------------------------------------------

# illustrate the income vs age of people in a scatter plot

# generate random integers (lower_range'inclusive', upper_range'exclusive', size of the array)
# age_array=np.random.randint(25,55,60)

# # Define an array containing salesamount values 
# income_array=np.random.randint(300000,700000,3000000)

# # create empty figure
# fig=go.Figure()

# # In go.Scatter we define the x-axis data, y-axis data and define the mode as markers with color of the marker as blue
# fig.add_trace(go.Scatter(x=age_array, y=income_array, mode='markers', marker=dict(color='blue')))

# # update the layout
# fig.update_layout(title='Economic Survey', xaxis_title='Age', yaxis_title='Income')
# fig.show()

# LINE PLOT --------------------------------------------------------------------------------------------------

# illustrate the sales of bicycles from Jan to August last year using a line chart

# Define an array containing numberofbicyclessold  
# numberofbicyclessold_array=[50,100,40,150,160,70,60,45]

# # Define an array containing months
# months_array=["Jan","Feb","Mar","April","May","June","July","August"]

# fig=go.Figure()

# fig.add_trace(go.Scatter(x=months_array, y=numberofbicyclessold_array, mode='lines', marker=dict(color='green')))
# fig.update_layout(title='Bicycle Sales', xaxis_title='Months', yaxis_title='Number of Bicycles Sold')
# fig.show()

# fig.write_html('bicycle_sales.html')


# BAR PLOT ---------------------------------------------------------------------------------------------------

# illustrate the average pass percentage of classes from grade 6 to grade 10

# Define an array containing scores of students 
# score_array=[80,90,56,88,95]
# # Define an array containing Grade names  
# grade_array=['Grade 6','Grade 7','Grade 8','Grade 9','Grade 10']

# # Use plotly express bar chart function px.bar.Provide input data, x and y axis variable, and title of the chart.
# # This will give average pass percentage per class
# fig = px.bar( x=grade_array, y=score_array, title='Pass Percentage of Classes') 
# fig.update_layout(title='Average Scores by Grade', xaxis_title='Grades', yaxis_title='Scores')
# fig.show()

# HISTOGRAM --------------------------------------------------------------------------------------------------

# .normal(mean, standard deviation, size)
# heights_array = np.random.normal(160, 11, 200)

# ## Use plotly express histogram chart function px.histogram.Provide input data x to the histogram
# fig = px.histogram(x=heights_array,title="Distribution of Heights")
# fig.show()

# BUBBLE PLOT ------------------------------------------------------------------------------------------------

# illustrate crime statistics of US cities with a bubble chart

#Create a dictionary having city,numberofcrimes and year as 3 keys
# crime_details = {
#     'City' : ['Chicago', 'Chicago', 'Austin', 'Austin','Seattle','Seattle'],
#     'Numberofcrimes' : [1000, 1200, 400, 700,350,1500],
#     'Year' : ['2007', '2008', '2007', '2008','2007','2008'],
# }
  
# # create a Dataframe object with the dictionary
# df = pd.DataFrame(crime_details)

# ## Group the number of crimes by city and find the total number of crimes per city
# bub_data = df.groupby('City')['Numberofcrimes'].sum().reset_index()
# print(bub_data)

# fig = px.scatter(bub_data, x="City", y="Numberofcrimes", size="Numberofcrimes",
#                  hover_name="City", title='Crime Statistics', size_max=60)
# fig.show()

# PIE CHART --------------------------------------------------------------------------------------------------

## Monthly expenditure of a family

# Random Data
# exp_percent= [20, 50, 10,8,12]
# house_holdcategories = ['Grocery', 'Rent', 'School Fees','Transport','Savings']

# # Values parameter will set values associated to the sector. 'exp_percent' feature is passed to it.
# # labels for the sector are passed to the `house hold categoris` parameter.
# fig = px.pie(values=exp_percent, names=house_holdcategories, title='Household Expenditure')
# fig.show()

# SUNBURST CHART --------------------------------------------------------------------------------------------

#Create a dictionary having a set of people represented by a character array and the parents of these characters represented in another
## array and the values are the values associated to the vectors.
# data = dict(
#     character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
#     parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
#     value=[10, 14, 12, 10, 2, 6, 6, 4, 4])

# fig = px.sunburst(
#     data,
#     names='character',
#     parents='parent',
#     values='value',
#     title="Family chart"
# )
# fig.show()

# ------------------------------------------- PART II PRACTICE -----------------------------------------------

# Read the airline data into pandas dataframe
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})

# Randomly sample 500 data points. n= # of samples, random state = random seed (train/test split) to be 42 so that we get same result.
data = airline_data.sample(n=500, random_state=42)

fig=go.Figure()

# print(airline_data.head())
# print(airline_data.shape)
# print(data.shape)
# print(data.columns)

# SCATTER PLOT -----------------------------------------------------------------------------------------------

# Departure Time changes with respect to airport distance

# fig=go.Figure()

# fig.add_trace(go.Scatter(x=data['Distance'], y=data['DepTime'], mode='markers', marker=dict(color='blue')))

# # update the layout
# fig.update_layout(title='Distance vs. Departure Time', xaxis_title='Distance', yaxis_title='Departure Time')
# fig.show()

# LINE PLOT --------------------------------------------------------------------------------------------------

# Average monthly arrival delay time

# Group the data by Month and compute average over arrival delay time.
# line_data = data.groupby('Month')['ArrDelay'].mean().reset_index()

# fig.add_trace(go.Scatter(x=line_data['Month'], y=line_data['ArrDelay'], mode='lines', marker=dict(color='blue')))

# # update the layout
# fig.update_layout(title='Average monthly arrival delay time', xaxis_title='Month', yaxis_title='Arrival Delay')
# fig.write_html('avg_monthly_arrival.html')

# BAR PLOT ---------------------------------------------------------------------------------------------------

# 

# Group the data by destination state and reporting airline. Compute total number of flights in each combination
# bar_data = data.groupby(['DestState'])['Flights'].sum().reset_index()
# bar_data = bar_data.sort_values(['Flights'], ascending=False, axis=0)
# # print(bar_data)

# fig = px.bar( x=bar_data['DestState'], y=bar_data['Flights']) 
# fig.update_layout(title='Most Traveled to States', xaxis_title='Dest State', yaxis_title='Flights')
# # fig.show()
# fig.write_html('dest_state.html')

# HISTOGRAM --------------------------------------------------------------------------------------------------

# Distiribution of arrival delay

# Set missing values to 0
# data['ArrDelay'] = data['ArrDelay'].fillna(0)

# fig = px.histogram(x=data['ArrDelay'],title="Distribution of Arrival Delays")
# fig.show()

# BUBBLE PLOT ------------------------------------------------------------------------------------------------

# Number of flights per airline

# Group the data by reporting airline and get number of flights
# bub_data = data.groupby('Reporting_Airline')['Flights'].sum().reset_index()

# fig = px.scatter(bub_data, x="Reporting_Airline", y="Flights", size="Flights",
#                  hover_name="Reporting_Airline", title='Flights per airline', size_max=60)
# fig.show()

# PIE CHART --------------------------------------------------------------------------------------------------

# Distance Group by Month

# Values parameter will set values associated to the sector. 'Month' feature is passed to it.
# labels for the sector are passed to the `names` parameter.
# fig = px.pie(data, values='Month', names='DistanceGroup', title='Distance Group Total by Month')
# fig.show()

# SUNBURST CHART --------------------------------------------------------------------------------------------

# 
fig = px.sunburst(data, path=['Month', 'DestStateName'], values='Flights',title='Flight Distribution Hierarchy')
fig.show()

# -----------------------------------------------------------------------------------------------------------

# airline_data.to_csv('airline_data.csv')