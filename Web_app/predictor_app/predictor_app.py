import flask
from flask import request
from predictor_api import make_prediction, feature_names

# Initialize the app
app = flask.Flask(__name__)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    x_input, predictions = make_prediction(request.args)
    return flask.render_template('predictor.html',
                                 x_input=x_input,
                                 feature_names=feature_names,
                                 prediction=predictions)


# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
if __name__ == '__main__':
    app.run(debug=True)

    # For public web serving:
    # app.run(host='0.0.0.0')


