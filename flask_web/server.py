import flask

from gevent import pywsgi 

app=flask.Flask(__name__)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
if __name__ == '__main__':
    app.run(port=4555, debug=True)
@app.route("/")
def index():
    f=open("index.html","rb")
    data = f.read()
    f.close()
    return data

app.run()
