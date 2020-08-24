import urllib.request
url = "http://127.0.0.1:5000"
p = "guangdong"
c = "shenzheng"
p=urllib.parse.quote(p)
c=urllib.parse.quote(c)
#print(p,c)
data="provice="+p+"&city="+c
resp=urllib.request.urlopen(url+"?"+data)
data=resp.read()
tml=data.decode()
print(html)