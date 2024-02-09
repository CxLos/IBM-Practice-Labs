# ------------------------------------------------- IMPORTS -----------------------------------------------

import pandas as pd
import requests
import flask
from flask import request, jsonify
from openpyxl import Workbook 

# ------------------------------------------------ FLASK API -----------------------------------------------

file = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\9. Capstone Project\Data\jobs.json'
j_data = pd.read_json(file)

jobs_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/module%201/Accessing%20Data%20Using%20APIs/jobs.json"

response = requests.get(jobs_url)
data = response.json()

# Function to get data based on key and value
def get_data(key, value, current):
    
    # Method if we are accessing data from url
    results = []
    for rec in current: # iterate over each job posting in the json
        if value in rec[key]:  # Check if entered value is contained in the specified key argument
            results.append(rec)
    return results

      # Method if we are accessing the data from a pandas dataframe
      # results = []
      # for rec in current.iterrows():  # Iterate over rows of DataFrame
      #     row = rec[1]  # Get row data
      #     if value in row[key]:  # Check if value is contained in specified key
      #         results.append(row.to_dict())  # Append row as dictionary to results
      # return results

# Flask app initialization
app = flask.Flask(__name__)

# data = None

# Route for the home page
@app.route('/', methods=['GET'])
def home():
    return '<h1>Welcome to Flask JOB search API</h1>'

# Route to get all job data
@app.route('/data/all', methods=['GET'])
def api_all():
    return jsonify(data)
      # return jsonify(j_data.to_dict(orient='records')) # method sourcing form local file 

# Route to display Python-related jobs
@app.route('/data/python', methods=['GET'])
def python_jobs():
    python_jobs = get_data('Key Skills', 'Python', data)
    return jsonify(python_jobs)

# Route to display jobs by specified location
@app.route('/data/location', methods=['GET'])
def location_jobs():
    location = request.args.get('location')  # Get location from request arguments
    location_jobs = get_data('Location', location, data)
    return jsonify(location_jobs)

# Route to display jobs in Los Angeles
@app.route('/data/location/los_angeles', methods=['GET'])
def los_angeles_jobs():
    los_angeles_jobs = get_data('Location', 'Los Angeles', data)
    return jsonify(los_angeles_jobs)

# Route to display Python jobs in Los Angeles
@app.route('/data/python/los_angeles', methods=['GET'])
def python_los_angeles_jobs():
    python_los_angeles_jobs = get_data('Key Skills', 'Python', data)
    python_los_angeles_jobs = get_data('Location', 'Los Angeles', python_los_angeles_jobs)
    return jsonify(python_los_angeles_jobs)

# Route to filter job data based on key-value pairs
@app.route('/data', methods=['GET'])
def api_id():
    res = None
    for req in request.args:
        if req == 'Job Title':
            key = 'Job Title'
        elif req == 'Job Experience Required':
            key = 'Job Experience Required'
        elif req == 'Key Skills':
            key = 'Key Skills'
        elif req == 'Role Category':
            key = 'Role Category'
        elif req == 'Location':
            key = 'Location'
        elif req == 'Functional Area':
            key = 'Functional Area'
        elif req == 'Industry':
            key = 'Industry'
        elif req == 'Role':
            key = 'Role'
        elif req == "id":
            key = "id"
        else:
            pass
        
        value = request.args[key]
        if res is None:
            res = get_data(key, value, data)
        else:
            res = get_data(key, value, res)
    return jsonify(res) #convert dictionary to json response obj

# Function to get the number of jobs for a specific technology
def get_number_of_jobs_T(technology):
    response = requests.get(jobs_url)
    data = response.json() # parse url to json
    count = sum(1 for job in data if technology in job['Key Skills'])
    return count

# Function to get number of jobs by location
def get_number_of_jobs_L(location):
    
  count = sum(1 for job in data if location == job['Location'])
  return count

# Function to get Python Jobs in Los Angeles
def get_number_of_jobs_PL():
    
  python_la_jobs = get_data('Key Skills', 'Python', data)
  python_la_jobs = get_data('Location', 'Los Angeles', python_la_jobs)
  count = len(python_la_jobs)
  print(f"Number of Python job openings in Los Angeles: {count}")
  return count

# Function to get jobs with a specific skill in a specific location
def get_number_of_jobs_SKILL_LOC(skill, location):
    skill_location_jobs = get_data('Key Skills', skill, data)
    skill_location_jobs = get_data('Location', location, skill_location_jobs)
    count = len(skill_location_jobs)
    print(f"{count} {skill} Job openings in {location}")
    return count

# Route to export Python jobs in Los Angeles to an Excel file
@app.route('/export/python/los_angeles', methods=['GET'])
def export_python_jobs_la():
    
    # Get Python jobs in Los Angeles
    python_jobs_la = get_data('Key Skills', 'Python', data)
    python_jobs_la = get_data('Location', 'Los Angeles', python_jobs_la)
    
    # Convert the filtered data to a DataFrame
    df = pd.DataFrame(python_jobs_la)
    
    # Specify the file path for the Excel file
    excel_file_path = 'python_jobs_la.xlsx'
    
    # Write the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)
    
    # Return a message indicating that the file has been created
    return f'Excel file for Python jobs in Los Angeles has been created: {excel_file_path}'


py_jobs = get_number_of_jobs_T("Python")
# print("Number of job openings that require Python:", py_jobs)

location = "New York"
ny_jobs = get_number_of_jobs_L(location)
# print(f"Number of job openings in {location}: {ny_jobs}")

location = "Los Angeles"
la_jobs = get_number_of_jobs_L(location)
# print(f"Number of job openings in {location}: {la_jobs}")

# get_number_of_jobs_PL()
# get_number_of_jobs_SKILL_LOC('Python','New York')

# SAVE TO EXCEL WITHOUT ROUTE

# Get Python jobs in Los Angeles
python_jobs_la = get_data('Key Skills', 'Python', data)
python_jobs_la = get_data('Location', 'Los Angeles', python_jobs_la)

# Convert the filtered data to a DataFrame
df = pd.DataFrame(python_jobs_la)

# Specify the file path for the Excel file
excel_file_path = 'python_jobs_la.xlsx'

# Write the DataFrame to an Excel file
# df.to_excel(excel_file_path, index=False)
# Print a message indicating that the file has been created
# print(f'Excel file for Python jobs in Los Angeles has been created: {excel_file_path}')


app.run()

# wb=Workbook()                        # create a workbook object
# ws=wb.active                         # use the active worksheet
# ws.append(['Country','Continent'])   # add a row with two columns 'Country' and 'Continent'
# ws.append(['Eygpt','Africa'])        # add a row with two columns 'Egypt' and 'Africa'
# ws.append(['India','Asia'])          # add another row
# ws.append(['France','Europe'])       # add another row
# wb.save("countries.xlsx")            # save the workbook into a file called countries.xlsx