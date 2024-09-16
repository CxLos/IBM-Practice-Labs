# IMPORTS ----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import folium
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import no_update
import requests
import csv
import json
import re
import os 
from PIL import Image
from IPython.display import IFrame
import flask
from flask import request, jsonify
from bs4 import BeautifulSoup 
import sqlite3

# Data ---------------------------------------------------------------------------------------------------

# path ='https://stackoverflow.blog/2019/04/09/the-2019-stack-overflow-developer-survey-results-are-in/'

# data = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\9. Capstone Project\Data\m1_survey_data.csv'

# df = pd.read_csv(data)

# 1. -------------------------------------- DATA COLLECTION -----------------------------------------------

# REQUESTS

# url='https://www.ibm.com/'
# r=requests.get(url)

# header=r.headers

# print('Date:', header['date']) #get date
# print('Content type:', header['Content-Type']) # 'text/html;charset=utf-8'
# print('Status code:', r.status_code) #200 is good
# print("request body:", r.request.body) # get the body element
# print('Encoding:', r.encoding) # check if utf-8
# print('Header:', header) # check response header
# print('Request Header:', r.request.headers) # retrieve html header section of page
# print(r.text[0:100]) # first 100 text

# url1='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/module%201/images/IDSNlogo.png'
# r1=requests.get(url1)
# path1=os.path.join(os.getcwd(),'image.png') # get path to image file
# header1=r1.headers

# print('Img file path:', path1) # get current file path
# print('Date:', header1['date']) #get date
# print('Content type:', header1['Content-Type']) # 'img/png'
# print('Status code:', r1.status_code) #200 is good
# print("Request body:", r1.request.body) # get the body element
# print('Encoding:', r1.encoding) # check if utf-8
# print('Header:', header1) # check response header
# print('Request Header:', r1.request.headers) # retrieve html header section of page
# print(r1.text[0:100]) # first 100 text

# create file to save image
# with open(path1,'wb') as f:
#     f.write(r1.content)

# Image.open(path) # open image

# url2 = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt'
# r2=requests.get(url2)
# path2=os.path.join(os.getcwd(),'example1.txt')
# header2=r2.headers

# print('Content type:', header2['Content-Type']) #text/plain

# with open(path2,'wb') as f:
#     f.write(r2.content)

# Get Requests with URL Parameters
    
# url_get='http://httpbin.org/get' # simple http request and response service
# payload={"name":"Joseph","ID":"123"} # pass in a payload
# r3=requests.get(url_get,params=payload)

# print(r3.url) # print out the url to site

# print("request body:", r3.request.body)
# print('Content type:', r3.headers['Content-Type']) # application/ json
# print(r3.status_code)
# print(r3.text)
# print(r3.json())
# print(r3.json()['args'])

# POST REQUESTS

# url_post='http://httpbin.org/post'
# r_post=requests.post(url_post,data=payload)

# print("POST request URL:",r_post.url )
# print("GET request URL:",r3.url)
# print("POST request body:",r_post.request.body)
# print("GET request body:",r3.request.body)
# print(r_post.json()['form'])

# COLLECTING JOB DATA USING API's

# api_url = "http://api.open-notify.org/astros.json"
# response = requests.get(api_url) 

# if response.ok:  
#     data = response.json() 

# astronauts = data.get('people')
# print("There are {} astronauts on ISS".format(len(astronauts)))
# print("And their names are :")

# for astronaut in astronauts:
#     print(astronaut.get('name'))

# print(data)  
# print(data.get('number'))

# 2. --------------------------------------- DATA WRANGLING -----------------------------------------------

# url = "http://www.ibm.com"

# # get the contents of the webpage in text format and store in a variable called data
# data  = requests.get(url).text 

# soup = BeautifulSoup(data,"html5lib")  # create a soup object using the variable 'data'

# for link in soup.find_all('a'):  # in html anchor/link is represented by the tag <a>
#     print('Links:', link.get('href'))

# for link in soup.find_all('img'):# in html image is represented by the tag <img>
#     print('Images:', link.get('src'))

# url1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"

# # get the contents of the webpage in text format and store in a variable called data
# data  = requests.get(url1).text
# soup = BeautifulSoup(data,"html5lib")
# table = soup.find('table') # in html table is represented by the tag <table>

#Get all rows from the table
# for row in table.find_all('tr'): # in html table row is represented by the tag <tr>
#     # Get all columns in each row.
#     cols = row.find_all('td') # in html a column is represented by the tag <td>
#     color_name = cols[2].getText() # store the value in column 3 as color_name
#     color_code = cols[3].getText() # store the value in column 4 as color_code
#     print("{}--->{}".format(color_name,color_code))

    #this url contains the data you need to scrape
# url2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/Programming_Languages.html"

# # get the contents of the webpage in text format and store in a variable called data
# data2  = requests.get(url2).text
# soup2 = BeautifulSoup(data2,"html5lib")
# table2 = soup2.find('table')
# print(table2)

# for row in table2.find_all('tr'):
   
#     cols = row.find_all('td') 
#     lang_name = cols[1].getText() 
#     avg_salary = cols[3].getText()
#     print("{}--->{}".format(lang_name, avg_salary))

    # Open a CSV file in write mode
# with open('programming_languages.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Write the header row
#     writer.writerow(['Language Name', 'Average Salary'])
    
#     # Iterate over each row in the table
#     for row in table2.find_all('tr'):
#         cols = row.find_all('td')
#         if len(cols) > 0:  # Ensure there are columns in the row
#             lang_name = cols[1].getText()
#             avg_salary = cols[3].getText()
#             print("{} ---> {}".format(lang_name, avg_salary))
            # writer.writerow([lang_name, avg_salary])

# print("Data has been saved to programming_languages.csv")

# dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m1_survey_data.csv"

# df = pd.read_csv(dataset_url)

# print(df.head())
# print('# of Rows:', df.shape[0]) # number of rows
# print('# of Columns:', df.shape[1]) # number of columns
# print(df.dtypes)

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m1_survey_data.csv")

# # print(df.head(10))
# # print(df.shape)
# # print(df.columns)

# # FINDING & REMOVING DUPLICATES
# duplicate_rows = df[df.duplicated()]
# duplicate_len = len(duplicate_rows)
# df_unique = df.drop_duplicates()

# # print("Shape of DataFrame before removing duplicates:", df.shape[0])
# # print("Shape of DataFrame after removing duplicates:", df_unique.shape[0])
# # print('Amt of duplicate rows:', duplicate_len)

# # MISSING & IMPUTING VALUES
# df_null = df_unique.isnull()
# df_null_count = df_null.value_counts()

# # print('Amt of null vlaues in df:', len(df_null_count))

# work_loc = df['WorkLoc']
# work_loc_len = len(df['WorkLoc'])
# # print("""Number of Rows in 'WorkLoc' column:""", work_loc_len)

# work_loc_null = df['WorkLoc'].isnull().sum()
# # print("""# of missing data in 'WorkLoc' column:""", work_loc_null)

# # Has values
# work_loc_not_null = df['WorkLoc'].notnull().sum()
# # print("""# of rows with data in 'WorkLoc' column:""", work_loc_not_null)

# # Most frequent work location
# most_common_work_loc1 = df['WorkLoc'].value_counts()
# most_common_work_loc2 = df['WorkLoc'].value_counts().sum()
# least_common_work_loc = df['WorkLoc'].value_counts().idxmin()
# most_common_work_loc = df['WorkLoc'].value_counts().idxmax()
# # print("WorkLoc value counts:", most_common_work_loc1)
# # print('Total:', most_common_work_loc2)
# # print("Most common work location:", most_common_work_loc)
# # print("Least common work location:", least_common_work_loc)

# # Replace missing values
# df['WorkLoc'].replace(np.nan, most_common_work_loc, inplace=True)
# work_loc_null = df['WorkLoc'].isnull().sum()
# # print("""# of missing data in 'WorkLoc' column after replacing:""", work_loc_null)

# # NORMALIZING DATA

# comp_mean = df["CompTotal"].value_counts().mean()
# comp_max = df['CompTotal'].max()
# df['CompTotal'].replace(np.nan, comp_mean, inplace=True)

# # print('CompTotal mean:', comp_mean)

# # comp_t = df['CompTotal']
# # comp_max = df['CompTotal'].max()
# # df['Normalized_CompTotal'] = comp_t / comp_max
# # norm_comp = df['Normalized_CompTotal']


# def normalize_compensation(row):
#     if row['CompFreq'] == 'Yearly':
#         return row['CompTotal'] / comp_max
#     elif row['CompFreq'] == 'Monthly':
#         return row['CompTotal'] * 12 / (comp_max * 12)
#     elif row['CompFreq'] == 'Weekly':
#         return row['CompTotal'] * 52 / (comp_max * 52)
#     else:
#         return comp_mean / comp_max
    
# # Apply the function to create the new column
# df['Normalized_CompTotal'] = df.apply(normalize_compensation, axis=1)
# median_norm = df['Normalized_CompTotal'].median()
# print(median_norm)

# df.sort_values(by='Normalized_CompTotal', ascending=False, axis=0, inplace=True)
# print(df.head(15))
# print(df[['Respondent','Normalized_CompTotal']].head(30))
# print(df[['Respondent','Normalized_CompTotal']].tail(10))

# print(norm_comp.head(10))

# Unique values
# comp_freq_unique = df['CompFreq'].unique()
# print('CompFreq categories:', comp_freq_unique)

# print(df['Respondent'].duplicated().vlaue_counts())

# 3. -------------------------------------- EXPLORATORY DATA -----------------------------------------------

# df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m2_survey_data.csv")

# DISTRIBUTION

# Plot the distribution curve for the column 'ConvertedComp'

# tr = df[['ConvertedComp', 'CompFreq']]

# plt.figure(figsize=(10, 6))
# plt.hist(tr, bins=30, color='skyblue', edgecolor='black') 
# sns.histplot(data=df, x='ConvertedComp', kde=True)
# plt.title('Distribution of Converted Compensation')
# plt.xlabel('Frequency')
# plt.ylabel('Converted Compensation')
# plt.grid(True)
# plt.show()

# Plot the Histogram

# plt.figure(figsize=(10, 6))
# plt.hist(tr, bins=30, color='skyblue', edgecolor='black') 
# plt.hist(data=df, x='ConvertedComp', kde=True)
# plt.title('Distribution of Converted Compensation')
# plt.xlabel('Frequency')
# plt.ylabel('Converted Compensation')
# plt.grid(True)
# plt.show()

# median = df['ConvertedComp'].median()
# print(median)

# Number of Men
# man = df['Gender'] == 'Man'
# print("Number of Men:", man.sum())

# Median ConvertedComp for women
# women = df[df['Gender'] == 'Woman']
# wmn_med = women['ConvertedComp'].median()
# print('Median ConvertedComp for Women:', wmn_med)

# Histogram for Age
# df_age = df['Age'].dropna()

# count, bin_edges = np.histogram(df_age, 7)
# print(count)
# print(bin_edges)

# plt.figure(figsize= (10, 6))
# sns.histplot(data=df, x='Age', kde=True, bins=bin_edges, linewidth=2, color='skyblue', edgecolor='black', alpha=0.7)
# df_age.plot(kind ='hist', 
#           figsize=(10, 6),
#           # bins=5,
#           alpha=0.6,
#           xticks=bin_edges,
#           edgecolor= 'black'
#           # color=['coral', 'darkslateblue', 'mediumseagreen']
#          )
# plt.title('Distribution of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.tight_layout() 
# plt.grid()

# OUTLIERS

# Find outliers using box-plot
# df_converted_comp = df['ConvertedComp']
# df_converted_comp.plot(kind='bar', figsize=(10,6))

# IQR:
# lower = df['ConvertedComp'].describe()['25%']
# upper = df['ConvertedComp'].describe()['75%']
# print('IQR:', lower, '-->', upper)

# How many outliers are there?
# Calculate the interquartile range (IQR)
# Q1 = df['ConvertedComp'].quantile(0.25)
# Q3 = df['ConvertedComp'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)

# Calculate the lower and upper bounds
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# Identify outliers
# outliers = df[(df['ConvertedComp'] < lower_bound) | (df['ConvertedComp'] > upper_bound)]

# Count the number of outliers
# num_outliers = len(outliers)

# print("Number of outliers in 'ConvertedComp' column:", num_outliers)

# Remove Outliers
# df_no_outliers = df[(df['ConvertedComp'] >= lower_bound) & (df['ConvertedComp'] <= upper_bound)]


# CORRELATION

# Find the correlation between 'Age' and all other numerical columns

# numeric_df = df.select_dtypes(include=['int64', 'float64'])
# age_corr = numeric_df.corr()['Age']
# print(age_corr)

# 4. ------------------------------------- DATA VISUALIZATION ----------------------------------------------

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m4_survey_data.sqlite"

filename = "m4_survey_data.sqlite"

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m2_survey_data.csv")

# conn = sqlite3.connect("m4_survey_data.sqlite") # open a database connection
# cur = conn.cursor()
# chunk = 100
# df.to_sql("SURVEY_DATA", conn, if_exists='replace', index=False, method="multi", chunksize=chunk)

# with open(filename, 'wb') as f:
#     f.write(response.content)
#     print("File downloaded successfully.")

# print all the tables names in the database
# QUERY = """
# SELECT name FROM sqlite_master WHERE type='table';
# """
# # the read_sql_query runs the sql query and returns the data as a dataframe
# pd.read_sql_query(QUERY,conn)

# df = pd.read_sql_query("""
# show tables;
# """, conn)

# print(df)
# conn.close()

# HISTOGRAM

# ccomp = df['ConvertedComp']
# plt.hist(ccomp)
# plt.show()
# ccomp.plot(kind='hist', figsize=(10,6))

# BOX PLOT
# plot a boxplot for age:
# df_age = df['Age']
# sns.boxplot(x=df_age)
# plt.show()
# df_age.plot(kind='box')


# SCATTER PLOT

# Create a scatter plot for Age & WorkWeekHrs

# df_scatter = df[['Age','WorkWeekHrs']]
# print(df_scatter.head())
# df_scatter.plot(kind='scatter', x='Age', y='WorkWeekHrs', figsize=(10, 6), color='darkblue')

# plt.title('Correlation between Age and Workweek Hours')
# plt.xlabel('Age')
# plt.ylabel('Workweek Hours')
# plt.show()

# BUBBLE PLOT

# Bubble Plot for WorkWeekHrs & CodeRevHrs using Age as bubble size

df_code_rev = df['CodeRevHrs'].dropna()
df_wwh = df['WorkWeekHrs'].dropna()
# df_code_wwh = df[['']]

df_bubble = df[['CodeRevHrs','WorkWeekHrs']].dropna()
# print(df_bubble.head())

norm_age = (df_bubble['Age'] - df_bubble['Age'].min()) / (df_bubble['Age'].max() - df_bubble['Age'].min())

# norm_code_rev = (df_bubble['CodeRevHrs'] - df_bubble['CodeRevHrs'].min()) / (df_bubble['CodeRevHrs'].max() - df['CodeRevHrs'].min())

# norm_wwh = (df_bubble['WorkWeekHrs'] - df_bubble['WorkWeekHrs'].min()) / (df_bubble['WorkWeekHrs'].max() - df['WorkWeekHrs'].min())

# WorkWeekHrs
ax0 = df_bubble.plot(kind='scatter',
                    x='Age',
                    y='CodeRevHrs',
                    figsize=(14, 8),
                    alpha=0.5,  # transparency
                    color='green',
                    s=norm_age * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                    )

# CodeRevHrs
ax1 = df_bubble.plot(kind='scatter',
                    x='Age',
                    y='WorkWeekHrs',
                    alpha=0.5,
                    color="blue",
                    s=norm_age * 2000 + 10,
                    ax=ax0
                    )

ax0.set_xlabel('CodeRevHrs')
ax0.set_ylabel('WorkWeekHrs')
ax0.set_title('CodeRevHrs vs. WorkWeekHrs with Age Bubble Size')


# PIE CHART

top_databases = df['DatabaseDesireNextYear'].value_counts().head(5)
# top_databases

# databases_worked = df['DatabaseWorkedWith'].value_counts().head(20)
# databases_worked

# Filter values containing 'sql' in the 'DatabaseWorkedWith' column and then count the occurrences
databases_worked_filtered = df[df['DatabaseWorkedWith'].str.contains('sql', case=False, na=False)]['DatabaseWorkedWith'].value_counts()
# databases_worked_filtered

databases_worked_counts = databases_worked_filtered.sum()
# databases_worked_counts

# Add up all the values
# total_counts = databases_worked_counts.sum()
# total_counts

top_lang = df['LanguageDesireNextYear'].value_counts().head(10)
top_lang

# df.columns

# df_db = df.groupby(['DatabaseDesireNextYear'], as_index=False)['DatabaseDesireNextYear'].value_counts()
# df_db.sort_values(['DatabaseDesireNextYear'], ascending=False, axis=0)

# # df_db = df.groupby(['LanguageDesireNextYear'], as_index=False)['LanguageDesireNextYear'].value_counts()
# # df_db.sort_values(['LanguageDesireNextYear'], ascending=False, axis=0)

# plt.figure(figsize=(8, 8))
# plt.pie(top_databases, labels=top_databases.index, autopct='%1.1f%%')
# # plt.pie(top_lang, labels=top_lang.index, autopct='%1.1f%%')
# plt.title('Top 5 Databases Respondents Wish to Learn Next Year')
# plt.show()

# STACKED CHARTS

# Create a stacked chart of median `WorkWeekHrs` and `CodeRevHrs` for the age group 30 to 35.

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Filter data for age group 30 to 35
df_age_group = df[(df['Age'] >= 30) & (df['Age'] <= 35)]

# Group by age and calculate median WorkWeekHrs and CodeRevHrs
df_age_group_median = df_age_group.groupby('Age')[['WorkWeekHrs', 'CodeRevHrs']].median()

# Plotting stacked bar chart
df_age_group_median.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Median WorkWeekHrs and CodeRevHrs for Age Group 30-35')
plt.xlabel('Age')
plt.ylabel('Median Hours')
plt.xticks(rotation=0)
plt.legend(title='Hours', loc='upper left')
plt.show()

# LINE CHART
# Plot the median ConvertedComp for all ages between 45 - 60

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Filter data for age group 30 to 35
df_age_group = df[(df['Age'] >= 45) & (df['Age'] <= 60)]

# Group by age and calculate median WorkWeekHrs and CodeRevHrs
df_converted = df_age_group.groupby('Age')[['ConvertedComp']].median()

# Plotting stacked bar chart
df_converted.plot(kind='line', stacked=True, figsize=(10, 6))

plt.title('Median WorkWeekHrs and CodeRevHrs for Age Group 30-35')
plt.xlabel('Age')
plt.ylabel('Median ConvertedComp')
# plt.xticks(rotation=0)
# plt.legend(title='Hours', loc='center')
plt.show()

# BAR CHART
# Create a horizontal bar chart using column 'MainBranch'

main_bar = df['MainBranch']

main_bar.value_counts().plot(kind='barh')

print(main_bar.value_counts())

# ----------------------------------------------- PRINTS ---------------------------------------------------

# print(df.head())
# print(df.columns)
# print(df['ConvertedComp'])
# df.to_csv('developer_surevey_result.csv')
# df_job.to_json('jobs.json')
# --------------------------------------------------------------------------------------------------------