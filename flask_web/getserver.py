import flask
app=flask.Flask(__name__)
@app.route("/")
def index():
    p=flask.request.args.get("provice")
    c=flask.request.args.get("city")
    print(p,c)
    return p+","+c
app.run()