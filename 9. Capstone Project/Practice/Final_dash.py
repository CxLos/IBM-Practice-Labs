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

# --------------------------------------------- DATA -----------------------------------------------------

data = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\9. Capstone Project\Data\m5_survey_data_technologies_normalised.csv'

data2 = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\9. Capstone Project\Data\m5_survey_data_demographics.csv'

df = pd.read_csv(data)
df2 = pd.read_csv(data2)

# -------------------------------------------------------------------------------------------------------

print(df.head(20))
# print(df2.head())