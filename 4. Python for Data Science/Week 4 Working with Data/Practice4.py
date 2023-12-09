import urllib.request 
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt'
filename = 'Example1.txt'
urllib.request.urlretrieve(url, filename)
from pyodide.http import pyfetch
import pandas as pd
from random import randint as rnd

exmp1 = './Week 4 Working with Data/Example1.txt'
exmp2 = './Week 4 Working with Data/Example2.txt'
exmp3 = './Week 4 Working with Data/Example3.txt'
file1 = open(exmp1, "r")
file2 = open(exmp2, "r")
file3 = open(exmp3, "r")
FileContent = file1.read()
FileContent2 = file2.read()
FileContent3 = file3.read()
Lines = ["Line A\n", "Line B\n", "Line C"]

# !curl Example1.txt https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt


# filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt"

# async def download(url, filename):

#     response = await pyfetch(url)

#     if response.status == 200:

#         with open(filename, "wb") as f:

#             f.write(await response.bytes())

#     await download(filename, "example1.txt")

# print("Downloaded")
# download(filename, "example1.txt")

print(file1.name)
print(file1.mode)
print(type(FileContent))
print(FileContent)

with open(example1, "r") as file1:
    FileContent = file1.read()
    print(FileContent)

with open(example1, "r") as file1:
    print(file1.read(4))
    print(file1.read(4))
    print(file1.read(7))
    print(file1.read(15))
    print(file1.read(16))
    print(file1.read(5))
    print(file1.read(9))

with open(example1, "r") as file1:
    print("first line: " + file1.readline())
    
with open(exmp2, 'w') as writefile:
    writefile.write("This is Line A\n")    
    writefile.write("This is line B\n")    
    

with open(exmp2, 'w') as writefile:
    for line in Lines:
        writefile.write(line)

with open(exmp2, 'r') as readfile:
    with open(exmp3, 'w') as writefile:
        for line in readfile:
            writefile.write(line)

with open(exmp2, 'r') as testwritefile:
    print(testwritefile.read())

with open(exmp2, 'a') as testwritefile:
    testwritefile.write("This is line C\n")
    testwritefile.write("This is line D\n")
    testwritefile.write("This is line E\n")

with open(exmp2, 'a+') as testwritefile:
    testwritefile.write("This is line F\n")

with open(exmp2, 'a+') as testwritefile:
    print("Initial Location: {}".format(testwritefile.tell()))

    data = testwritefile.read()
    if (not data):  #empty strings return false in python
            print('Read nothing') 
    else: 
            print(testwritefile.read())
            
    testwritefile.seek(0,0) # move 0 bytes from beginning.
    
    print("\nNew Location : {}".format(testwritefile.tell()))
    data = testwritefile.read()
    if (not data): 
            print('Read nothing') 
    else: 
            print(data)
    
    print("Location after read: {}".format(testwritefile.tell()) )

with open(exmp2, 'r+') as testwritefile:
    testwritefile.seek(0,0) #write at beginning of file
   
    testwritefile.write("Line 1" + "\n")
    testwritefile.write("Line 2" + "\n")
    testwritefile.write("Line 3" + "\n")
    testwritefile.write("Line 4" + "\n")
    testwritefile.write("finished\n")
    testwritefile.seek(0,0)
    print(testwritefile.read())

with open(exmp2, 'r+') as testwritefile:
    testwritefile.seek(0,2) #write at beginning of file
   
    testwritefile.write("Line 1" + "\n")
    testwritefile.write("Line 2" + "\n")
    testwritefile.write("Line 3" + "\n")
    testwritefile.write("Line 4" + "\n")
    testwritefile.write("finished\n")
    testwritefile.truncate()
    testwritefile.seek(0,0)
    print(testwritefile.read())

