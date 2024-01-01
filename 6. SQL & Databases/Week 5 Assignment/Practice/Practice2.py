
import csv, sqlite3
import pandas
from pyodide.http import pyfetch
import urllib.request 
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

con = sqlite3.connect("Chicago_Schools.db")
cur = con.cursor()

# Absolute Path
file = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\6. SQL & Databases\Week 5 Assignment\CSV\ChicagoPublicSchools.csv'

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoPublicSchools.csv'

df = pandas.read_csv(file)
df.to_sql("CHICAGO_PUBLIC_SCHOOLS_DATA", con, if_exists='replace', index=False, method="multi", chunksize=50)
df = pandas.read_sql_query("select * from CHICAGO_PUBLIC_SCHOOLS_DATA limit 50;", con)

# Retrieve list of all tables in your schema
# df4 = pandas.read_sql_query("select name from sqlite_master where type='table';", con)

# Column Count/ Info bout table
df4 = pandas.read_sql_query("PRAGMA table_info('CHICAGO_PUBLIC_SCHOOLS_DATA');", con)
column_count = len(df4)

# Use PRAGMA table_info to get information about the columns
cur.execute(f"PRAGMA table_info('CHICAGO_PUBLIC_SCHOOLS_DATA');")
columns_info = cur.fetchall()
# Extract the column names from the result
column_names = [column[1] for column in columns_info]

# Displaying the list of tables and their creation time
# for table in tables:
#     table_name, create_sql, tbl_name = table
#     print(f"Table: {table_name}, Creation Time: {create_sql}")

print(df)
# print(df4)
# print(df.columns)
# print(os.getcwd())
# print(column_count)
# print(column_names)
con.close()