# -*- coding: utf-8 -*-

import argparse
import os
from flask import Flask, request, jsonify
from flasgger import Swagger
from machine_learning.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from werkzeug.exceptions import BadRequest, InternalServerError
from intent_classifier import IntentClassifier
from log_handler import file_handler, log_level
from prometheus_flask_exporter import PrometheusMetrics



app = Flask(__name__)
# App logger configuration
app.logger.addHandler(file_handler)
app.logger.setLevel(log_level)
# Swagger configuration
app.config['SWAGGER'] = {
    'title': 'Intent Classifier API',
    'description': 'API for Intent Classification',
    'uiversion': 3
}
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.3')
#swagger =
Swagger(app)
model = None
@app.route('/api/example', methods=['GET'])
def example_endpoint():
    return jsonify({"message": "Success"})
# Global variable to hold the model

@app.route('/ready')
def ready():

    if model and model.is_ready():
        app.logger.info("ready")
        return 'OK', 200
    else:
        app.logger.info("server not ready")
        return 'Not ready', 423

@app.route('/intent', methods=['POST'])
def intent():
    try:
        data = request.get_json()
        if not data:
            app.logger.error('Request body is missing.')
            return jsonify({"label": "BODY_MISSING", "message": "Request body is missing."}), 400
        if 'text' not in data:
            app.logger.error('Request body is missing "text" field.')
            return jsonify({"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}), 400
        if not isinstance(data['text'], str):
            app.logger.error('Request body has invalid "text" field.')
            return jsonify({"label": "INVALID_TYPE", "message": "\"text\" is not a string."}), 400
        if not data['text']:
            app.logger.error('Request body has empty "text" field.')
            return jsonify({"label": "TEXT_EMPTY", "message": "\"text\" is empty."}), 400
    except BadRequest:
        app.logger.error('Invalid JSON format.')
        return jsonify({"label": "BAD_REQUEST", "message": "Invalid JSON format."}), 400

    # Predict the intent
    predictions = model.predict(data['text'])
    app.logger.info('User entry: %s', data['text'])
    app.logger.info('Predicted intent: %s', predictions)
    # Return the response
    return jsonify({"intents": predictions}), 200

@app.errorhandler(500)
def internal_error(error):
    app.logger.error('Server encountered an internal error: %s', error)
    return jsonify({
        "label": "INTERNAL_ERROR",
        "message": str(error) or "Internal server error."
    }), 500
@app.route('/')
def index():
    app.logger.info('Info level log')
    app.logger.warning('Warning level log')
    return "Check your logs!"

if __name__ == '__main__':
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
        arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
        args = arg_parser.parse_args()
        app.logger.info(f"Loading model.{args.model}")

        # Initialize and load the model
        model = IntentClassifier(args.model)
        model.load(args.model)
        app.logger.info('Application started successfully')
        if not model.is_ready():
            raise Exception("Model is not ready.")
    except Exception as e:
        print(e)
        # Raise an internal server error
        raise InternalServerError()
    # Start the Flask app
    app.run(host='0.0.0.0', port=args.port)