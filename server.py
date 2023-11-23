# -*- coding: utf-8 -*-

import argparse
import os
from flask import Flask, request, jsonify
from flasgger import Swagger
from machine_learning.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from werkzeug.exceptions import BadRequest, InternalServerError
from intent_classifier import IntentClassifier

app = Flask(__name__)
Swagger(app)
model = None
@app.route('/api/example', methods=['GET'])
def example_endpoint():
    return jsonify({"message": "Success"})
# Global variable to hold the model

@app.route('/ready')
def ready():
    if model and model.is_ready():
        return 'OK', 200
    else:
        return 'Not ready', 423

@app.route('/intent', methods=['POST'])
def intent():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"label": "BODY_MISSING", "message": "Request body is missing."}), 400
        if 'text' not in data:
            return jsonify({"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}), 400
        if not isinstance(data['text'], str):
            return jsonify({"label": "INVALID_TYPE", "message": "\"text\" is not a string."}), 400
        if not data['text']:
            return jsonify({"label": "TEXT_EMPTY", "message": "\"text\" is empty."}), 400
    except BadRequest:
        return jsonify({"label": "BAD_REQUEST", "message": "Invalid JSON format."}), 400

    # Predict the intent
    predictions = model.predict(data['text'])

    # Return the response
    return jsonify({"intents": predictions}), 200

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "label": "INTERNAL_ERROR",
        "message": str(error) or "Internal server error."
    }), 500

if __name__ == '__main__':
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
        arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
        args = arg_parser.parse_args()
        print(args.model)

        # Initialize and load the model
        model = IntentClassifier(args.model)
        model.load(args.model)
        print("model loaded")
        print(model.predict("Hi I am running and will got to new york"))
        if not model.is_ready():
            raise Exception("Model is not ready.")
    except Exception as e:
        print(e)
        # Raise an internal server error
        raise InternalServerError()
    # Start the Flask app
    app.run(host='0.0.0.0', port=args.port)