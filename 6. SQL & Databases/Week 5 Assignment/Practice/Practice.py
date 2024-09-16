
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

# file = './IBM Practice Labs/ChicagoPublicSchools.csv'
file = r'C:/Users/CxLos/OneDrive/Documents/IBM Data Analyst Professional Certificate/IBM Practice Labs/ChicagoPublicSchools.csv'
absolute_path = os.path.abspath(file)
script_dir = os.path.dirname(os.path.abspath(__file__))
# print('Script Directory:', script_dir)
# print(f"Absolute path of the file: {absolute_path}")

con = sqlite3.connect("Chicago_Schools.db")
cur = con.cursor()

df = pandas.read_csv(url)
df2 = pandas.read_csv(url2)
df3 = pandas.read_csv(url3)

df.to_sql("CENSUS_DATA", con, if_exists='replace', index=False,method="multi")
# df2.to_sql("CHICAGO_CRIME_DATA", con, if_exists='replace', index=False, method="multi")
df3.to_sql("CHICAGO_PUBLIC_SCHOOLS_DATA", con, if_exists='replace', index=False, method="multi", chunksize=50)

df1 = pandas.read_sql_query("select * from CENSUS_DATA limit 10;", con)
# df22 = pandas.read_sql_query("select * from CHICAGO_CRIME_DATA limit 10;", con)
df33 = pandas.read_sql_query("select * from CHICAGO_PUBLIC_SCHOOLS_DATA limit 7;", con)

# Retrieve list of all tables in your schema
# df4 = pandas.read_sql_query("select name from sqlite_master where type='table';", con)

# Count columns
# df4 = pandas.read_sql_query("select count(name) from PRAGMA_TABLE_INFO('CHICAGO_PUBLIC_SCHOOLS_DATA');", con)

# List of Columns and their type
# df4 = pandas.read_sql_query("select name, type, length(type) from PRAGMA_TABLE_INFO('CHICAGO_PUBLIC_SCHOOLS_DATA');", con)

# 1. Count Elementary schools
df4 = pandas.read_sql_query("select count(*) from CHICAGO_PUBLIC_SCHOOLS_DATA where [Elementary, Middle, or High School]='ES';", con)

# 2. Highest Safety Score
# df4 = pandas.read_sql_query("select name_of_school, max(safety_score) from CHICAGO_PUBLIC_SCHOOLS_DATA;", con)

# 3. Schools with highest safety score
# df4 = pandas.read_sql_query("select name_of_school, safety_score from CHICAGO_PUBLIC_SCHOOLS_DATA where safety_score = (select max(safety_score) from CHICAGO_PUBLIC_SCHOOLS_DATA limit 10);", con)

# # 4. Top 10 schools with highest average student attendance
# df4 = pandas.read_sql_query("select name_of_school, average_student_attendance from CHICAGO_PUBLIC_SCHOOLS_DATA order by Average_Student_Attendance desc nulls last limit 10 ;", con)

# # 5. 5 schools with the lowest avg student attendance in ASC order of attendance
# df4 = pandas.read_sql_query("select name_of_school, average_student_attendance from CHICAGO_PUBLIC_SCHOOLS_DATA order by average_student_attendance asc nulls last limit 20;", con)

# # 6. Remove '%' symbol from above problem

# df4 = pandas.read_sql_query("select name_of_school, replace(average_student_attendance, '%', '') from CHICAGO_PUBLIC_SCHOOLS_DATA order by average_student_attendance asc nulls last limit 10;", con)

# # 7. Schools with avg attendance lower than 70%
# df4 = pandas.read_sql_query("select name_of_school, average_student_attendance from CHICAGO_PUBLIC_SCHOOLS_DATA where average_student_attendance < 70 order by average_student_attendance desc limit 20;", con)

# # 8. Total college enrollment for each community area
# df4 = pandas.read_sql_query("select community_area_name, sum(college_enrollment_rate__) as total_enrollment from CHICAGO_PUBLIC_SCHOOLS_DATA group by community_area_name order by total_enrollment desc limit 20;", con)

# # 9. 5 Community areas with least total College enrollment in ASC order
# df4 = pandas.read_sql_query("select community_area_name, sum(college_enrollment_rate__) as total_enrollment from CHICAGO_PUBLIC_SCHOOLS_DATA group by community_area_name order by total_enrollment asc limit 10;", con)

# # 10. 5 schools with lowest safety score
# df4 = pandas.read_sql_query("select name_of_school, safety_score from CHICAGO_PUBLIC_SCHOOLS_DATA order by safety_score asc nulls last limit 10;", con)

# # 11. Hardship index for community with college enrollment of 4368
# df4 = pandas.read_sql_query(""" select CD.community_area_name, hardship_index from CENSUS_DATA CD, CHICAGO_PUBLIC_SCHOOLS_DATA CPS 
# where CD.community_area_number = CPS.community_area_number and college_enrollment > 4368; """, con)

# # 12. Hardship index for the community area that has highest value for college enrollment
# df4 = pandas.read_sql_query("select CD.community_area_name, hardship_index from CENSUS_DATA CD, CHICAGO_PUBLIC_SCHOOLS_DATA CPS where CD.community_area_number = CPS.community_area_number and college_enrollment = (select max(college_enrollment) from CHICAGO_PUBLIC_SCHOOLS_DATA);", con)

# df4 = pandas.read_sql_query("select community_area_name, college_enrollment from CHICAGO_PUBLIC_SCHOOLS_DATA order by college_enrollment desc limit 20;", con)

# print(df1)
# print(df22)
# print(df1)
# print(df1.columns)
print(df4)

con.close()

# Street_Address', 'City', 'State', 'ZIP_Code', 'Phone_Number', 'Link',
#        'Network_Manager', 'Collaborative_Name',
#        'Adequate_Yearly_Progress_Made_', 'Track_Schedule',
#        'CPS_Performance_Policy_Status', 'CPS_Performance_Policy_Level',
#        'HEALTHY_SCHOOL_CERTIFIED', 'Safety_Icon', 'SAFETY_SCORE',
#        'Family_Involvement_Icon', 'Family_Involvement_Score',
#        'Environment_Icon', 'Environment_Score', 'Instruction_Icon',
#        'Instruction_Score', 'Leaders_Icon', 'Leaders_Score', 'Teachers_Icon',
#        'Teachers_Score', 'Parent_Engagement_Icon', 'Parent_Engagement_Score',
#        'Parent_Environment_Icon', 'Parent_Environment_Score',
#        'AVERAGE_STUDENT_ATTENDANCE', 'Rate_of_Misconducts__per_100_students_',
#        'Average_Teacher_Attendance',
#        'Individualized_Education_Program_Compliance_Rate', 'Pk_2_Literacy__',
#        'Pk_2_Math__', 'Gr3_5_Grade_Level_Math__', 'Gr3_5_Grade_Level_Read__',
#        'Gr3_5_Keep_Pace_Read__', 'Gr3_5_Keep_Pace_Math__',
#        'Gr6_8_Grade_Level_Math__', 'Gr6_8_Grade_Level_Read__',
#        'Gr6_8_Keep_Pace_Math_', 'Gr6_8_Keep_Pace_Read__',
#        'Gr_8_Explore_Math__', 'Gr_8_Explore_Read__', 'ISAT_Exceeding_Math__',
#        'ISAT_Exceeding_Reading__', 'ISAT_Value_Add_Math',
#        'ISAT_Value_Add_Read', 'ISAT_Value_Add_Color_Math',
#        'ISAT_Value_Add_Color_Read', 'Students_Taking__Algebra__',
#        'Students_Passing__Algebra__', '9th Grade EXPLORE (2009)',
#        '9th Grade EXPLORE (2010)', '10th Grade PLAN (2009)',
#        '10th Grade PLAN (2010)', 'Net_Change_EXPLORE_and_PLAN',
#        '11th Grade Average ACT (2011)', 'Net_Change_PLAN_and_ACT',
#        'College_Eligibility__', 'Graduation_Rate__',
#        'College_Enrollment_Rate__', 'COLLEGE_ENROLLMENT',
#        'General_Services_Route', 'Freshman_on_Track_Rate__', 'X_COORDINATE',
#        'Y_COORDINATE', 'Latitude', 'Longitude', 'COMMUNITY_AREA_NUMBER',
#        'COMMUNITY_AREA_NAME', 'Ward', 'Police_District', 'Location'],
#       dtype='object')