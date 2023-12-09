
from bs4 import BeautifulSoup
import requests
import pandas as pd

# html="<!DOCTYPE html><html><head><title>Page Title</title></head><body><h3><b id='boldest'>Lebron James</b></h3><p> Salary: $ 92,000,000 </p><h3> Stephen Curry</h3><p> Salary: $85,000, 000 </p><h3> Kevin Durant </h3><p> Salary: $73,200, 000</p></body></html>"

# soup = BeautifulSoup(html, 'html5lib')
# pretty = soup.prettify()
# tag_object = soup.h3
# tag_child = tag_object.b
# parent_tag = tag_child.parent
# sibling1 = tag_object.next_sibling
# sibling2 = sibling1.next_sibling
# tag_string= tag_child.string

# print(pretty)
# # print(tag_object)
# print(type(tag_object))
# print(tag_child)
# print(parent_tag)
# print(sibling1)
# print(sibling2)

# TABLE

# table="<table><tr><td id='flight'>Flight No</td><td>Launch site</td> <td>Payload mass</td></tr><tr> <td>1</td><td><a href='https://en.wikipedia.org/wiki/Florida'>Florida<a></td><td>300 kg</td></tr><tr><td>2</td><td><a href='https://en.wikipedia.org/wiki/Texas'>Texas</a></td><td>94 kg</td></tr><tr><td>3</td><td><a href='https://en.wikipedia.org/wiki/Florida'>Florida<a> </td><td>80 kg</td></tr></table>"

# table_bs = BeautifulSoup(table, 'html5lib')
# tpretty = table_bs.prettify()

# table_rows=table_bs.find_all('tr')
# first_row =table_rows[0]
# first_child = first_row.td
# list_input=table_bs .find_all(name=["tr", "td"])
# flight = table_bs.find_all(id="flight")
# wiki = table_bs.find_all(href="https://en.wikipedia.org/wiki/Florida")
# troo = table_bs.find_all(href=True)
# bold = table_bs.find_all(id="boldest")
# fl = table_bs.find_all(string="Florida")

# for i,row in enumerate(table_rows):
#     print("row",i,"is",row)

# for i,row in enumerate(table_rows):
#     print("row",i)
#     cells=row.find_all('td')
#     for j,cell in enumerate(cells):
#         print('colunm',j,"cell",cell)

# print(tpretty)
# print(table_bs)
# print(table_rows)
# print(first_row)
# print(first_child)
# print(type(first_row))
# print(list_input)
# print(flight)
# print(wiki)
# print(troo)
# print(bold)
# print(fl)

# two_tables="<h3>Rocket Launch </h3><p><table class='rocket'><tr><td>Flight No</td><td>Launch site</td> <td>Payload mass</td></tr><tr><td>1</td><td>Florida</td><td>300 kg</td></tr><tr><td>2</td><td>Texas</td><td>94 kg</td></tr><tr><td>3</td><td>Florida </td><td>80 kg</td></tr></table></p><p><h3>Pizza Party  </h3><table class='pizza'><tr><td>Pizza Place</td><td>Orders</td> <td>Slices </td></tr><tr><td>Domino's Pizza</td><td>10</td><td>100</td></tr><tr><td>Little Caesars</td><td>12</td><td >144 </td></tr><tr><td>Papa John's </td><td>15 </td><td>165</td></tr>"

# two_tables_bs= BeautifulSoup(two_tables, 'html.parser')
# table = two_tables_bs.find("table")
# pizza = two_tables_bs.find("table",class_='pizza')

# # print(two_tables)
# # print(table)
# print(pizza)

# 

# url = "http://www.ibm.com"

# data = requests.get(url).text
# soup = BeautifulSoup(data,"html5lib")

# for link in soup.find_all('a',href=True):  
#     print(link.get('href'))

# for link in soup.find_all('img'):# in html image is represented by the tag <img>
#     print(link)
#     print(link.get('src'))

# # print(soup)

# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"

# data  = requests.get(url).text
# soup = BeautifulSoup(data,"html5lib")
# table = soup.find('table')

# for row in table.find_all('tr'): # in html table row is represented by the tag <tr>
#     # Get all columns in each row.
#     cols = row.find_all('td') # in html a column is represented by the tag <td>
#     color_name = cols[2].string # store the value in column 3 as color_name
#     color_code = cols[3].text # store the value in column 4 as color_code
#     print("{}--->{}".format(color_name,color_code))

# # print(table)

# 

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"

tables = pd.read_html(url)

# print(tables)
print(tables[0])