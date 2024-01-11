import pandas as pd
import numpy as np
import asyncio
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

async def main():
    file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"

    await download(file_path, "laptops.csv")
    file_name = "laptops.csv"

    df = pd.read_csv(file_name)
    print(df.head(5))

# asyncio.run(main())

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
file_name="laptops.csv"

df = pd.read_csv(file_path, header=None)

headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

df.columns = headers
df.columns

df = df.replace('?', np.NaN)

print(df.head(10))
print(df.dtypes)
print(df.describe())
print(df.info())