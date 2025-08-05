import csv, sqlite3
import pandas
from pyodide.http import pyfetch
import urllib.request 
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
import os

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCensusData.csv'
url2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCrimeData.csv'
url3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoPublicSchools.csv'
url4 = 'https://data.cityofchicago.org/resource/jcxq-k9xf.csv'

# File route that doesn't work for some reason
# file = './IBM Practice Labs/ChicagoPublicSchools.csv'
# file = 'C:/Users/CxLos/OneDrive/Documents/IBM Data Analyst Professional Certificate/IBM Practice Labs/ChicagoPublicSchools.csv'
# absolute_path = os.path.abspath(file)
# print(f"Absolute path of the file: {absolute_path}")

con = sqlite3.connect("Chicago_Schools.db")
cur = con.cursor()

df = pandas.read_csv(url)
df.to_sql("CENSUS_DATA", con, if_exists='replace', index=False,method="multi")
df1 = pandas.read_sql_query("select * from CENSUS_DATA limit 10;", con)

df2 = pandas.read_csv(url2)
df2.to_sql("CHICAGO_CRIME_DATA", con, if_exists='replace', index=False, method="multi")
df22 = pandas.read_sql_query("select * from CHICAGO_CRIME_DATA limit 10;", con)

df3 = pandas.read_csv(url3)
df3.to_sql("CHICAGO_PUBLIC_SCHOOLS_DATA", con, if_exists='replace', index=False, method="multi", chunksize=50)
df33 = pandas.read_sql_query("select * from CHICAGO_PUBLIC_SCHOOLS_DATA limit 10;", con)

# 1. Total number of crimes recorded in the crime table
# df4 = pandas.read_sql_query("select count(case_number) from CHICAGO_CRIME_DATA;", con)

# # 2. Community area names & numbers with per capita income less than 11,000
# df4 = pandas.read_sql_query("select Community_area_number, community_area_name, per_capita_income from CENSUS_DATA where per_capita_income < 11000 ORDER BY per_capita_income;", con)

# # 3. All case numbers for crimes involving minors
# df4 = pandas.read_sql_query(""" select case_number from CHICAGO_CRIME_DATA where description like "%minor%"; """, con)

# # 4. Kidnapping crimes involving a child
# df4 = pandas.read_sql_query("""select CASE_NUMBER,PRIMARY_TYPE, DESCRIPTION from CHICAGO_CRIME_DATA WHERE PRIMARY_TYPE = "KIDNAPPING" limit 10;""", con)

# # 5. Types of crimes recorded at schools
# df4 = pandas.read_sql_query(""" select DISTINCT PRIMARY_TYPE from CHICAGO_CRIME_DATA WHERE LOCATION_DESCRIPTION IN ('SCHOOL, PUBLIC, GROUNDS', 'SCHOOL, PUBLIC, BUILDING', 'SCHOOL, PRIVATE, BUILDING') limit 20; """, con)

# # 6. type of schools along with average safety score for each type.
# df4 = pandas.read_sql_query(""" select "Elementary, Middle, or High School", AVG(safety_score) as avg_safety_score from CHICAGO_PUBLIC_SCHOOLS_DATA GROUP BY "Elementary, Middle, or High School" limit 20; """, con)

# # 7. 5 community areas with highest % of households below poverty line.
# df4 = pandas.read_sql_query(""" select Community_Area_Name, Percent_Households_Below_Poverty from CENSUS_DATA ORDER BY Percent_Households_Below_Poverty DESC LIMIT 5; """, con)

# # 8. Most crime prone community areas. Show only community area number.
# df4 = pandas.read_sql_query(""" select community_area_number, community_area_name, COUNT(COMMUNITY_AREA_NUMBER) AS CRIMES from CHICAGO_CRIME_DATA GROUP BY COMMUNITY_AREA_NUMBER ORDER BY CRIMES DESC LIMIT 1; """, con)

# # 9. Subquery to find name of community area with highest hardship index.
# df4 = pandas.read_sql_query(""" select Community_area_name, MAX(hardship_index) from CENSUS_DATA; """, con)

# # 10. Subquery to determine community area name with highest number of crimes
df4 = pandas.read_sql_query(
""" 
    SELECT community_area_name
    FROM CENSUS_DATA
    WHERE COMMUNITY_AREA_NUMBER IN (
        SELECT CCD.COMMUNITY_AREA_NUMBER
        FROM CHICAGO_CRIME_DATA CCD
        GROUP BY CCD.COMMUNITY_AREA_NUMBER
        ORDER BY COUNT(CCD.COMMUNITY_AREA_NUMBER) DESC
        LIMIT 10);
""", con)

# unique = df2["LOCATION_DESCRIPTION"].unique()

# print(unique)
print(df4)
# print(df22)
# print(df1.columns)
# print(df22.columns)
# print(df33.columns)
con.close()