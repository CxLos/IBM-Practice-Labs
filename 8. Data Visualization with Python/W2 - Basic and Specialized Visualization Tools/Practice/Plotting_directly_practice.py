# IMPORTS --------------------------------------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
from io import StringIO
# from js import fetch
import io

# FETCHING DATA ----------------------------------------------------------------------------------------------

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv"

# resp = await fetch(URL)
# text = io.BytesIO((await resp.arrayBuffer()).to_py())

response = requests.get(URL)
data = StringIO(response.text)
df_can = pd.read_csv(data)

# Set index to country
df_can.set_index('Country', inplace=True)

# Store years in a variable
# years = list(map(str, range(1980, 2014)))
years = np.arange(1980,2014)

# LINE PLOT --------------------------------------------------------------------------------------------------



# SCATTER PLOT ----------------------------------------------------------------------------------------------



# BAR PLOT --------------------------------------------------------------------------------------------------



# HISTOGRAM ------------------------------------------------------------------------------------------------



# PIE CHART ------------------------------------------------------------------------------------------------



# SUB-PLOTTING ---------------------------------------------------------------------------------------------



# PRINTS ----------------------------------------------------------------------------------------------------

print(df_can.head())
print('Years List:', years)
print('data dimensions:', df_can.shape)

# check for latest version of Matplotlib
print('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# -----------------------------------------------------------------------------------------------------------