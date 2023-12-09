
import requests
import os 
from PIL import Image
from IPython.display import IFrame

# url='https://www.ibm.com/'
url='https://www.httpbin.org/'
gurl='https://www.httpbin.org/get'
r = requests.get(url)
post = "http://httpbin.org/post"
payload = {"name": "Fran", "ID": "2"}
rpost = requests.post(post, data=payload)
rget = requests.get(gurl, params=payload)

header = r.headers
body = r.request.body
text = r.text
encode = r.encoding
# json = r.json()



print(r.status_code)
print(header['date'])
print(rget.url)
# print(body)
# print(text[0:100])
# print(json)
# print(rpost)
print(rget.text)
print(rget.json()['args'])
print("POST request body:", rpost.request.body)
print("GET request body:", r.request.body)
# print(rpost.json()['form'])