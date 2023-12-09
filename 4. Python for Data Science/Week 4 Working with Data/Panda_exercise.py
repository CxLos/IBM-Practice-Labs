import pandas as pd
# import piplite
# from pyodide.http import pyfetch

# dataFrame = {'Numbers': [10, 20, 30, 40, 50], 'Colors': ['red', 'blue', 'green', 'yellow', 'white']}

# x = {'Name': ['Rose','John', 'Jane', 'Mary'], 'ID': [1, 2, 3, 4], 'Department': ['Architect Group', 'Software Group', 'Design Team', 'Infrastructure'], 'Salary':[100000, 80000, 50000, 60000]}

# s = pd.Series(dataFrame)
# df = pd.DataFrame(dataFrame)
# df = pd.DataFrame(x)
# x = df[['Colors']]
# z = df[['Department','Salary','ID']]

# print(s)
# print(s[2])
# print(s.iloc[2])
# print(s[0:4])
# dataFrame.head()
# print(df)
# print(type(x))
# print(z)
# print(x)

# roster = {'Student': ['David', 'Samuel', 'Terry', 'Evan',], 'Age': [27, 24, 22, 32], 'Country': ['UK', 'Canadia', 'China', 'USA',], 'Course': ['Python', 'Data Structures', 'Machine Learning', 'Web Development'], 'Marks': [85, 72, 89, 76]}

# dd = pd.DataFrame(roster)
# print(dd)
# print(type(dd))

# filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/TopSellingAlbums.csv"

# filename = './Week 4 Working with Data/TopSellingAlbums.csv'
# filename2 = './Week 4 Working with Data/TopSellingAlbums.xlsx'
# fr = open(filename, 'r')
# frx = open(filename2, 'r')
# fro = fr.read()
# fro2 = frx.read()

# async def download(url, filename):
#     # response = await pyfetch(url)
#     # if response.status == 200:
#         with open(filename, "wb") as f:
#             f.write(filename, "TopSellingAlbums.csv")
            # f.write(await response.bytes())
        # await download(filename, "TopSellingAlbums.csv")

# download(filename, "TopSellingAlbums.csv")
# df = pd.read_csv(filename)
# dff = pd.DataFrame(filename)
# print(fro)

# filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/TopSellingAlbums.csv"

filename = './Week 4 Working with Data/TopSellingAlbums.csv'
filename2 = './Week 4 Working with Data/TopSellingAlbums.xlsx'
df = pd.read_csv(filename)
df2 = pd.read_excel(filename2)
x = df[['Length']]
x1 = df[['Artist']]
y = df[['Artist', 'Length', 'Genre']]
new_index=['a','b','c','d','e','f','g','h']
df3 = df
df3.index = new_index

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
    await download(filename, "TopSellingAlbums.csv")


print(df)
# print(df.head())
# print(x)
# print(x1)
# print(y)
# print(type(x))
# print('Locations: \n', df.iloc[0,0])
# print(df.iloc[1,0])
# print(df.iloc[0,2])
print(df.iloc[1,2])
print(df.loc[2,2])
# print(df.loc[0, 'Artist'])
# print(df.loc[1, 'Artist'])
# print(df.loc[0, 'Released'])
# print(df.loc[1, 'Released'])
# print(df.iloc[0:2, 0:3])
# print(df.loc[0:2, 'Artist':'Released'])