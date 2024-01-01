
# Import sqlite3
import sqlite3

import pandas as pd

# Connect to sqlite
# Connection Object
conn = sqlite3.connect('INSTRUCTOR_db')

# Cursor Object
# In order to execute SQL statements and fetch results from SQL queries, we will need to use a database cursor. Call conn.cursor() to create the Cursor:
cursor_obj = conn.cursor()

# Drop the table if already exists.
cursor_obj.execute("DROP TABLE IF EXISTS INSTRUCTOR")

# Creating table
table = """ create table IF NOT EXISTS INSTRUCTOR(
    ID INTEGER PRIMARY KEY NOT NULL, 
    FNAME VARCHAR(20), 
    LNAME VARCHAR(20), 
    CITY VARCHAR(20), 
    CCODE CHAR(2));"""
 
cursor_obj.execute(table)
 
print("Table is Ready")
# print(table)

# Inserting data to the table
cursor_obj.execute(
  '''insert into INSTRUCTOR 
  values (1, 'Rav', 'Ahuja', 'TORONTO', 'CA')''')

cursor_obj.execute('''insert into INSTRUCTOR values (2, 'Raul', 'Chong', 'Markham', 'CA'), 
                   (3, 'Hima', 'Vasudevan', 'Chicago', 'US')''')

# Query Data from table

# statement = '''SELECT * FROM INSTRUCTOR'''
# cursor_obj.execute(statement)

# print("All the data:")
# output_all = cursor_obj.fetchall()
# for row_all in output_all:
#   print(row_all)

# If you want to fetch few rows from the table we use fetchmany(numberofrows) and mention the number how many rows you want to fetch

# statement = '''SELECT * FROM INSTRUCTOR'''
# cursor_obj.execute(statement)

# print("Some of the data:")
# output_many = cursor_obj.fetchmany(2) 
# for row_many in output_many:
#   print(row_many)

# Fetch only FNAME from the table

# statement = '''SELECT FNAME FROM INSTRUCTOR'''
# cursor_obj.execute(statement)

# print("All First Names:")
# output_column = cursor_obj.fetchall()
# for fetch in output_column:
#   print(fetch)

# Update
# query_update='''update INSTRUCTOR set CITY='MOOSETOWN' where FNAME="Rav"'''
# cursor_obj.execute(query_update)

# statement = '''SELECT * FROM INSTRUCTOR'''
# cursor_obj.execute(statement)
  
# print("Updated data:")
# output1 = cursor_obj.fetchmany(4)
# for row in output1:
#   print(row)

#retrieve the query results into a pandas dataframe

df = pd.read_sql_query("select * from instructor;", conn)

# print(df)
# print(df.LNAME[0])
# print(df.shape)

# Close the connection
conn.close()