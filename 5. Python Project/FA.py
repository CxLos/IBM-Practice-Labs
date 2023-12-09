
from datetime import datetime
import yfinance as yf
import pandas as pd
import requests
import urllib.request
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# GRAPH FUNCTION

# Tesla Graph

def make_graph1(stock_data, revenue_data, stock):
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    
    stock_data_specific = stock_data[stock_data.Date <= '2021-06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()

# Gamestop Graph    
    
def make_graph2(stock_data, revenue_data, stock):
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    
    stock_data_specific = stock_data[stock_data.index <= '2021-06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.index, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    # pyo.iplot(fig)
    fig.show()

#  TESLA

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm'

html_data = requests.get(url).text
df = pd.read_html(html_data)
soup = BeautifulSoup(html_data, 'html.parser')
pretty = soup.prettify()

TSLA = "TSLA"
Tesla = yf.Ticker("TSLA")
Tesla_data = Tesla.history(period='ytd')
T_reset = Tesla_data.reset_index(inplace=True)
table = soup.find_all("table")
tables = str(table)
tesla_revenue = pd.read_html(tables)

if tesla_revenue:
    tesla_revenue_df = tesla_revenue[1]
    tesla_revenue_df.columns = ["Date", "Revenue"]
    # The line `tesla_revenue_df.dropna(inplace=True)` is removing any rows in the DataFrame
    # `tesla_revenue_df` that contain missing values (NaN) and modifying the DataFrame in place.
    tesla_revenue_df.dropna(inplace=True)
    tesla_revenue_df['Revenue'] = tesla_revenue_df['Revenue'].replace('[\\$,]','', regex=True).astype(float)
    # tesla_revenue_df['Revenue'] = tesla_revenue_df['Revenue'].replace('[\$,]', '', regex=True)
    # tesla_revenue_df['Revenue'] = pd.to_numeric(tesla_revenue_df['Revenue'], errors='coerce')
    tesla_revenue_df = tesla_revenue_df[tesla_revenue_df['Revenue'] != ""]
    # Now 'tesla_revenue_df' is a DataFrame containing the data from the HTML table
else:
    print("No tables found in the HTML content.")

# make_graph1(Tesla_data, tesla_revenue_df, TSLA)
print(tesla_revenue_df)
# print(Tesla_data)

#  GAMESTOP

urlg = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html'

html_data2 = requests.get(urlg).text
df2 = pd.read_html(html_data2)
soup2 = BeautifulSoup(html_data2, 'html.parser')
pretty2 = soup2.prettify

GME = "GME"
gamestop = yf.Ticker("GME")
gamestop_shares = gamestop.history(period='ytd')
gamestop_shares_specific = gamestop_shares[gamestop_shares.index <= '2021-06-14']
gamestop_reset = gamestop_shares.reset_index(inplace=True)

table2 = soup2.find_all('table')
tables2 = str(table2)
gamestop_revenue = pd.read_html(tables2)

if gamestop_revenue:
  gamestop_revenue_df = gamestop_revenue[1]
  gamestop_revenue_df.columns = ["Date", "Revenue"]
  gamestop_revenue_df['Revenue'] = gamestop_revenue_df['Revenue'].replace('[\\$,]', '', regex=True).astype(float)
  # gamestop_revenue_df['Date'] = gamestop_revenue_df['Date'].astype(int)
  # gamestop_revenue_df['Revenue'] = gamestop_revenue_df['Revenue'].replace('[\$,]', '', regex=True)
  # gamestop_revenue_df['Revenue'] = pd.to_numeric(gamestop_revenue_df['Revenue'], errors='coerce')


make_graph1(gamestop_shares, gamestop_revenue_df, GME)
# print(gamestop_revenue_df)


