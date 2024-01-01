import csv, sqlite3
import pandas
from pyodide.http import pyfetch
import urllib.request 
import requests
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://data.cityofchicago.org/resource/jcxq-k9xf.csv'
# file = './Practice/jcxq-k9xf.csv'

con = sqlite3.connect("socioeconomics.db")
cur = con.cursor()

df = pandas.read_csv(url)
df.to_sql("chicago_socioeconomic_data", con, if_exists='replace', index=False,method="multi")
df = pandas.read_sql_query("select * from chicago_socioeconomic_data limit 5;", con)
# print(df)

# df2 = pandas.read_sql_query("select * from chicago_socioeconomic_data limit 5;", con)
df2 = pandas.read_sql_query("select count (*) from chicago_socioeconomic_data;", con)
# df2 = pandas.read_sql_query("select count (hardship_index) from chicago_socioeconomic_data where hardship_index > 50;", con)
# df2 = pandas.read_sql_query("select community_area_name from chicago_socioeconomic_data where hardship_index = 98;", con)
# df2 = pandas.read_sql_query("select community_area_name from chicago_socioeconomic_data where per_capita_income_ > 60000;", con)

# Chart

# income_vs_hardship = pandas.read_sql_query("select per_capita_income_, hardship_index from chicago_socioeconomic_data;", con)
# plot = sns.jointplot(x='per_capita_income_', y='hardship_index', data=income_vs_hardship)
# plt.show()

print(df2)
# print(pl)

con.close()