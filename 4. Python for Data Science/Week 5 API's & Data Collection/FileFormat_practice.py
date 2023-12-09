
import xml.etree.ElementTree as ET
import urllib.request
from PIL import Image 
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

# url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/addresses.csv'

# df = pd.read_csv(url,header=None)
# df.columns =['First Name', 'Last Name', 'Location ', 'City','State','Area Code']
# fName = df["First Name"]

# # print(df)
# print(fName)



# df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
# df = df.transform(func = lambda x : x + 10)
# result = df.transform(func = ['sqrt'])

# print(df)
# print(result)

#  JSON

# person = {
#     'first_name' : 'Mark',
#     'last_name' : 'abc',
#     'age' : 27,
#     'address': {
#         "streetAddress": "21 2nd Street",
#         "city": "New York",
#         "state": "NY",
#         "postalCode": "10021-3100"
#     }
# }

# json_object = json.dumps(person, indent = 4) 

# with open('person.json', 'w') as f: 
#     json.dump(person, f)
  
# Writing to sample.json 
# with open("sample.json", "w") as outfile: 
#     outfile.write(json_object) 

# with open('sample.json', 'r') as openfile: 
  
#     # Reading from json file 
#   json_object = json.load(openfile) 

# print(person)
# print(json_object)

# XML

urllib.request.urlretrieve("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/file_example_XLSX_10.xlsx", "sample.xlsx")

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/Sample-employee-XML-file.xml'

r = requests.get(url).text

df = pd.read_excel(r)

employee = ET.Element('employee')
details = ET.SubElement(employee, 'details')
first = ET.SubElement(details, 'firstname')
second = ET.SubElement(details, 'lastname')
third = ET.SubElement(details, 'age')
first.text = 'Shiv'
second.text = 'Mishra'
third.text = '23'

# create a new XML file with the results
mydata1 = ET.ElementTree(employee)
# myfile = open("items2.xml", "wb")
# myfile.write(mydata)
with open("new_sample.xml", "wb") as files:
    mydata1.write(files)

tree = ET.parse(r)

root = tree.getroot()
columns = ["firstname", "lastname", "title", "division", "building","room"]

datatframe = pd.DataFrame(columns = columns)

for node in root: 

    firstname = node.find("firstname").text

    lastname = node.find("lastname").text 

    title = node.find("title").text 
    
    division = node.find("division").text 
    
    building = node.find("building").text
    
    room = node.find("room").text
    
    datatframe = datatframe.append(pd.Series([firstname, lastname, title, division, building, room], index = columns), ignore_index = True)

# print(r)
print(df)
# print(mydata1)
# print(datatframe)

urllib.request.urlretrieve("https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg", "dog.jpg")

img = Image.open('dog.jpg') 
  
# Output Images 
# print(display(img))

# print(i)

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/diabetes.csv"

df = pd.read_csv(path)
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   

# print("The first 5 rows of the dataframe") 
# print(df.head(5))
# print(df.info())
# print(df.describe())
# print(missing_data)
# print(df.dtypes)

# labels= 'Not Diabetic','Diabetic'
# plt.pie(df['Outcome'].value_counts(),labels=labels,autopct='%0.02f%%')
# plt.legend()
# plt.show()

# def add(x):
#   y = x + x
#   return(y)

# print(add(1))

# a = np.array([0,1,0,1,0])
# b = np.array([1,0,1,0,1])

# print(a/b)