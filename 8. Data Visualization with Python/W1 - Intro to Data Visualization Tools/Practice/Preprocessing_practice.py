import matplotlib.pyplot as plt 
import numpy as np  # useful for many scientific computing in Python
import pandas as pd

df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

# df_can.set_index('Country', inplace=True)

# df_can.reset_index()

# print(df_can.head())
# print(df_can.loc['Albania'])
# print(df_can.index.tolist())
# print(df_can.info(verbose=False))

# plt.plot(5,5,'o')
# plt.show()

# df_can.loc['Haiti', years].plot(kind = 'line')
df_can.to_excel('Canada_Immigration.xlsx')