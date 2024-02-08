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
import os 
from PIL import Image
from IPython.display import IFrame

# Data ---------------------------------------------------------------------------------------------------

# path ='https://stackoverflow.blog/2019/04/09/the-2019-stack-overflow-developer-survey-results-are-in/'

# data = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\9. Capstone Project\Data\m1_survey_data.csv'

# df = pd.read_csv(data)

# 1. -------------------------------------- DATA COLLECTION ---------------------------------------------

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

# Image.open(path) #open image

# url2 = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt'
# r2=requests.get(url2)
# path2=os.path.join(os.getcwd(),'example1.txt')
# header2=r2.headers

# print('Content type:', header2['Content-Type']) #text/plain

# with open(path2,'wb') as f:
#     f.write(r2.content)

# Get Requests with URL Parameters
    
url_get='http://httpbin.org/get' # simple http request and response service
payload={"name":"Joseph","ID":"123"} # pass in a payload
r3=requests.get(url_get,params=payload)

# print(r3.url) # print out the url to site

# print("request body:", r3.request.body)
# print('Content type:', r3.headers['Content-Type']) # application/ json
# print(r3.status_code)
# print(r3.text)
# print(r3.json())
# print(r3.json()['args'])

# POST REQUESTS

url_post='http://httpbin.org/post'
r_post=requests.post(url_post,data=payload)

print("POST request URL:",r_post.url )
print("GET request URL:",r3.url)
print("POST request body:",r_post.request.body)
print("GET request body:",r3.request.body)
print(r_post.json()['form'])

# 2. --------------------------------------- DATA WRANGLING -----------------------------------------------



# 3. -------------------------------------- EXPLORATORY DATA -----------------------------------------------



# 4. ------------------------------------- DATA VISUALIZATION ----------------------------------------------


# ----------------------------------------------- PRINTS ---------------------------------------------------

# print(df.head())

# df.to_csv('developer_surevey_result.csv')
# --------------------------------------------------------------------------------------------------------