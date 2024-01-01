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

# 1-1. List the school names, community names and average attendance for communities with hardship index of 98

# df4 = pandas.read_sql_query(
#   """ SELECT 
#   P.NAME_OF_SCHOOL, P.AVERAGE_STUDENT_ATTENDANCE, P.COMMUNITY_AREA_NAME, C.COMMUNITY_AREA_NAME, C.HARDSHIP_INDEX 
#   FROM CHICAGO_PUBLIC_SCHOOLS_DATA P 
#   LEFT JOIN CENSUS_DATA C ON P.COMMUNITY_AREA_NAME=C.COMMUNITY_AREA_NAME 
#   AND C.HARDSHIP_INDEX = 20;"""
#   , con)

# df4 = pandas.read_sql_query(
#   """SELECT 
#   P.NAME_OF_SCHOOL, P.AVERAGE_STUDENT_ATTENDANCE, P.COMMUNITY_AREA_NAME, C.COMMUNITY_AREA_NAME, C.HARDSHIP_INDEX FROM CENSUS_DATA C 
#   LEFT JOIN CHICAGO_PUBLIC_SCHOOLS_DATA P ON C.COMMUNITY_AREA_NAME=P.COMMUNITY_AREA_NAME 
#   WHERE C.HARDSHIP_INDEX = 98;"""
#   , con)

# df4 = pandas.read_sql_query(
#   "SELECT C.COMMUNITY_AREA_NAME, C.HARDSHIP_INDEX FROM CENSUS_DATA C LEFT JOIN CHICAGO_PUBLIC_SCHOOLS_DATA S ON S.COMMUNITY_AREA_NAME = C.COMMUNITY_AREA_NAME WHERE S.COMMUNITY_AREA_NAME IS NULL;"
#   , con)

# 1-2. List all crimes that took place at a school. include case number crime type, and community name

# df4 = pandas.read_sql_query(
#   """ SELECT R.CASE_NUMBER, R.PRIMARY_TYPE, R.LOCATION_DESCRIPTION, C.COMMUNITY_AREA_NAME FROM CHICAGO_CRIME_DATA R LEFT JOIN CENSUS_DATA C ON R.COMMUNITY_AREA_NUMBER=C.COMMUNITY_AREA_NUMBER WHERE R.LOCATION_DESCRIPTION LIKE 'SCHOOL%'; """
# , con)

# 2-1. Create a view that displays.

con.execute("DROP VIEW CHICAGO_SCHOOLS")

con.execute( """ CREATE VIEW CHICAGO_SCHOOLS AS 
            SELECT 
            NAME_OF_SCHOOL AS School_Name, 
            SAFETY_ICON AS Safety_Rating, 
            FAMILY_INVOLVEMENT_ICON AS Family_Rating, 
            ENVIRONMENT_ICON AS Environment_Rating, 
            INSTRUCTION_ICON AS Instruction_Rating, 
            LEADERS_ICON AS Leaders_Rating,
            TEACHERS_ICON AS Teachers_Rating
            FROM CHICAGO_PUBLIC_SCHOOLS_DATA
            LIMIT 10; """ )

df4 = pandas.read_sql_query(
  """ SELECT * FROM CHICAGO_SCHOOLS """
, con)

# df4 = pandas.read_sql_query(
#   """ SELECT school_name FROM CHICAGO_SCHOOLS """
# , con)

# 3-1. Write a query to create or replace a stored procedure called UPDATE_LEADERS_SCORE that takes in_school_ID parameter as integer and in_Leader_Score paramater as integer. use #SET TERMINATOR @

# con.execute("""  
#     CREATE PROCEDURE UPDATE_LEADERS_SCORE
#   """)

# 3-2. Inside the previous stored procedure, write a SQL statement to update the Leaders_Score field in the public schools table for the school identified by in_school_id to the value in the in_leader_score parameter

# con.execute("""  
#     CREATE PROCEDURE UPDATE_LEADERS_SCORE
#       (IN SCHOOL_ID INT, IN LEADERS_RATING INT)
#   """)

# 3-3. Inside the stored procedure, write a SQL IF statement to update the leaders_icon field in the public schools table for the school identified by in_school_id using the following information.

# con.execute(
#   """
#   CREATE PROCEDURE UPDATE_LEADERS_SCORE
#     (IN SCHOOL_ID INT, IN LEADERS_RATING INT)  

#       BEGIN

#         IF LEADERS_RATING >79 THEN
#           UPDATE CHICAGO_SCHOOLS
#           SET LEADERS_RATING = 'Very Strong'
#           WHERE ID = SCHOOL_ID;
#         ELSEIF LEADERS_RATING > 59 AND LEADERS_RATING < 80 THEN
#           UPDATE CHICAGO_SCHOOLS
#           SET LEADERS_RATING = 'STRONG'
#           WHERE ID = SCHOOL_ID;
#         ELSEIF LEADERS_RATING > 39 AND LEADERS_RATING < 60 THEN
#           UPDATE CHICAGO_SCHOOLS
#           SET LEADERS_RATING = 'Average'
#           WHERE ID = SCHOOL_ID;
#         ELSEIF LEADERS_RATING > 19 AND LEADERS_RATING <40 THEN
#           UPDATE CHICAGO_SCHOOLS
#           SET LEADERS_RATING = 'Weak'
#           WHERE ID = SCHOOL_ID;
#         ELSEIF LEADERS_RATING < 20 THEN
#           UPDATE CHICAGO_SCHOOLS
#           SET LEADERS_RATING = 'Very Weak'
#           WHERE ID = SCHOOL_ID;

#       END @
#             """)

# con.execute(
#     """
# CREATE TRIGGER update_leaders_score
# AFTER INSERT ON CHICAGO_PUBLIC_SCHOOLS_DATA
# FOR EACH ROW
# BEGIN
#     UPDATE CHICAGO_PUBLIC_SCHOOLS_DATA
#     SET LEADERS_ICON =
#         CASE
#             WHEN NEW.LEADERS_ICON > 79 THEN 'Very Strong'
#             WHEN NEW.LEADERS_ICON > 59 AND NEW.LEADERS_ICON < 80 THEN 'STRONG'
#             WHEN NEW.LEADERS_ICON > 39 AND NEW.LEADERS_ICON < 60 THEN 'Average'
#             WHEN NEW.LEADERS_ICON > 19 AND NEW.LEADERS_ICON < 40 THEN 'Weak'
#             WHEN NEW.LEADERS_ICON < 20 THEN 'Very Weak'
#             ELSE 'Unknown'
#         END
#     WHERE SCOOL_ID = NEW.SCHOOL_ID;
# END;
#     """
# )

# 3-4. Write a query to call the stored procedure, passing a valid school ID and a leader score of 50, to check that the procedure works as expected

# def update_leaders_score(school_id, leaders_icon):
#     con.execute(
#         """
#         UPDATE CHICAGO_PUBLIC_SCHOOLS_DATA
#         SET LEADERS_ICON =
#             CASE
#                 WHEN ? > 79 THEN 'Very Strong'
#                 WHEN ? > 59 AND ? < 80 THEN 'STRONG'
#                 WHEN ? > 39 AND ? < 60 THEN 'Average'
#                 WHEN ? > 19 AND ? < 40 THEN 'Weak'
#                 WHEN ? < 20 THEN 'Very Weak'
#             END
#         WHERE SCHOOL_ID = ?;
#         """,
#         (leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, school_id)
#     )
#     con.commit()
#     (print("Leader Score Entered =", leaders_icon))
#     (print('Updated Table:'))

# update_leaders_score(610334, 80)  

# df4 = pandas.read_sql_query (""" select school_id, LEADERS_ICON from CHICAGO_PUBLIC_SCHOOLS_DATA where school_id = 610334 """, con)

# 4-1 update stored procedure definition. Add a generic ELSE clause to the IF statement that rolls back to the current work if the score did not fit any of the preceding categories.
# Hint: you can add an ELSE clause to the IF statement which will only run if none of the previous conditions have been met.

# def update_leaders_score(school_id, leaders_icon):
    
#   con.execute("""
#         UPDATE CHICAGO_PUBLIC_SCHOOLS_DATA
#           SET LEADERS_ICON =
#               CASE
#                   WHEN ? > 79 THEN 'Very Strong'
#                   WHEN ? > 59 AND ? < 80 THEN 'STRONG'
#                   WHEN ? > 39 AND ? < 60 THEN 'Average'
#                   WHEN ? > 19 AND ? < 40 THEN 'Weak'
#                   WHEN ? < 20 THEN 'Very Weak'
#                 ELSE 'Unknown'
#               END
#           WHERE SCHOOL_ID = ?;
#           """,
#           (leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, school_id)
#         )

#   (print("Leader Score Entered =", leaders_icon))

# update_leaders_score(610334, 80)  

# df4 = pandas.read_sql_query (""" select school_id, LEADERS_ICON from CHICAGO_PUBLIC_SCHOOLS_DATA where school_id = 610334; """, con)

# 4-2. Update stored procedure definition again. Add a statement to commit the current unit of work at the end of the procedure.
# Hint: Remember that as soon as any code inside the IF/ELSE IF/ELSE statement completes, processing resumes after the END IF, so you can add your commit code there.
# Write and run one query to check that the updated stored procedure works as expected when you use a valid score of 38.
# # Write and Run another query to check that the updated stored procedure works as expected when you use an invalid score of 101.

# def update_leaders_score(school_id, leaders_icon):
    
#   con.execute("""
#     BEGIN
#     DECLARE EXIT HANDLER FOR SQLEXCEPTION
#     BEGIN
#         ROLLBACK;
#         RESIGNAL;
#     END;

#     START TRANSACTION;
#         UPDATE CHICAGO_PUBLIC_SCHOOLS_DATA
#           SET LEADERS_ICON =
#               CASE
#                   WHEN ? > 79 THEN 'Very Strong'
#                   WHEN ? > 59 AND ? < 80 THEN 'STRONG'
#                   WHEN ? > 39 AND ? < 60 THEN 'Average'
#                   WHEN ? > 19 AND ? < 40 THEN 'Weak'
#                   WHEN ? < 20 THEN 'Very Weak'
#                 ELSE 'Unknown'
#               END
#           WHERE SCHOOL_ID = ?;
#           """,
#           (leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, leaders_icon, school_id)
#         )
#   (print("Leader Score Entered =", leaders_icon))

# update_leaders_score(610334, 80)  

# df4 = pandas.read_sql_query (""" select school_id, LEADERS_ICON from CHICAGO_PUBLIC_SCHOOLS_DATA where school_id = 610334; """, con)

print(df4)
# print(df22)
# print(unique)
# print(df1.columns)
# print(df22.columns)
# print(df33.columns)
con.close()