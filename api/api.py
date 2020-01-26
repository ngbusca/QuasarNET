import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/instructions', methods=['GET'])
def home():
    return "<h1>instructions here</h1><p> QuasarNET rest-api instructions.</p>"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')