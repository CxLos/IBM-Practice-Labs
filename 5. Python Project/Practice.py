IBM_cloud_feature_code = '180d34b177b6e14d74f00fa0fb2087b1'

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

today = datetime.today().strftime('%Y-%M-%D')
yversion = yf.__version__
# apple_file = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/apple.json'
apple_file = './5. Python Project/AAPL.txt'
apple_file1 = open(apple_file, "r")
apple_file2 = apple_file1.read()

amd_file = './5. Python Project/AMD.txt'
amd_open = open(amd_file, "r")
amd = amd_open.read()
amd_ticker = yf.Ticker("AMD")
amd_shares = amd_ticker.history(period="ytd")
amd_reset = amd_shares.reset_index(inplace=True)
amd_dividends = amd_ticker.dividends 

apple = yf.Ticker("AAPL")
apple_share_price_data = apple.history(period="ytd")
areset = apple_share_price_data.reset_index(inplace=True)
aplot = apple_share_price_data.plot(x="Date", y="Open")
adividends = apple.dividends.plot()

msft = yf.Ticker("MSFT")
msft_data = msft.history(period="ytd")
mHead = msft_data.head()

with open(apple_file) as json_file:
  apple_info = json.load(json_file)

with open(amd_file) as json_file:
  amd_info = json.load(json_file)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm'

def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
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
  

# print(mHead)
# print(msft_data)
# print(apple.info)
# print(apple_file2['country'])
# print(apple_info['country'])
# print(apple_share_price_data)
# print(apple_share_price_data.head())
# print(aplot)
# print(adividends)
# print(amd)
# print(amd_info)
# print(amd_info['country'])
# print(amd_info['sector'])
# print(amd_country)
# print(amd_ticker)
print(amd_shares.head())
# print(amd_reset)
# print(yversion)