import urllib.request
url="http://127.0.0.1:5000"
resp=urllib.request.urlopen(url)
data=resp.read()
html=data.decode()
print(html)
