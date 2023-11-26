# -*- coding: utf-8 -*-

import argparse
import os
import time
from flask import Flask, request, jsonify
from flasgger import Swagger
from werkzeug.exceptions import BadRequest, InternalServerError
from intent_classifier import IntentClassifier
from log_handler import file_handler, log_level
from prometheus_flask_exporter import PrometheusMetrics, Counter, Histogram
from flask_sqlalchemy import SQLAlchemy
import json


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
Swagger(app)

# Prometheus Metrics configuration
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.3')

# Custom Prometheus Metrics
REQUEST_COUNT = Counter(
    'request_count', 'App Request Count',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'Request latency',
    ['endpoint']
)

model = None
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requests.db'
db = SQLAlchemy(app)

class IntentRequest(db.Model):
    """
    Represents a request to the intent classifier API.
    """
    id = db.Column(db.Integer, primary_key=True)
    request_text = db.Column(db.String, nullable=False)
    response_data = db.Column(db.String, nullable=False)
    response_time = db.Column(db.Float, nullable=False)

class IntentLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    intent = db.Column(db.String(500))  # or db.Text if you expect larger text
    response = db.Column(db.String(500))  # or db.Text

    def __repr__(self):
        return f'<IntentLog {self.id}, Intent: {self.intent}>'

with app.app_context():
    app.logger.info("Creating database tables...")
    db.create_all()
    app.logger.info(db.Model.metadata.tables)
    app.logger.info("Database tables created.")
    db.metadata


@app.route('/api/example', methods=['GET'])
def example_endpoint():
    """
    Example endpoint

    ---
    tags:
        - Example
    responses:
        200:
            description: Successful response.
            schema:
                type: object
                properties:
                    message:
                        type: string

    """
    REQUEST_COUNT.labels('GET', '/api/example', '200').inc()
    return jsonify({"message": "Success"})

@app.route('/ready')
def ready():
    """
    Check if the server is ready to accept requests.

    ---
    tags:
        - Health
    responses:
        200:
            description: Server is ready.
        423:
            description: Server is not ready.

    """
    if model and model.is_ready():
        app.logger.info("ready")
        REQUEST_COUNT.labels('GET', '/ready', '200').inc()
        return 'OK', 200
    else:
        app.logger.info("server not ready")
        REQUEST_COUNT.labels('GET', '/ready', '423').inc()
        return 'Not ready', 423

@app.route('/intent', methods=['POST'])
def intent():
    """
        Intent Classification

        ---
        tags:
          - Intent
        parameters:
          - name: text
            in: body
            description: Text to classify intent. 
            required: true
            schema:
              type: string
        responses:
          200:
            description: Successful classification.
            schema:
              type: object
              properties:
                intents:
                  type: array
                  items:
                    type: string
          400:
            description: Invalid request or missing data.
            schema:
              type: object
              properties:
                label:
                  type: string
                message:
                  type: string
        """
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            REQUEST_COUNT.labels('POST', '/intent', '400').inc()
            app.logger.error('Request body is missing.')
            return jsonify({"label": "BODY_MISSING", "message": "Request body is missing."}), 400
        if 'text' not in data:
            REQUEST_COUNT.labels('POST', '/intent', '400').inc()
            app.logger.error('Request body is missing "text" field.')
            return jsonify({"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}), 400
        if not isinstance(data['text'], str):
            REQUEST_COUNT.labels('POST', '/intent', '400').inc()
            app.logger.error('Request body has invalid "text" field.')
            return jsonify({"label": "INVALID_TYPE", "message": "\"text\" is not a string."}), 400
        if not data['text']:
            REQUEST_COUNT.labels('POST', '/intent', '400').inc()
            app.logger.error('Request body has empty "text" field.')
            return jsonify({"label": "TEXT_EMPTY", "message": "\"text\" is empty."}), 400

        # Predict the intent
        predictions = model.predict(data['text'])
        app.logger.info('User entry: %s', data['text'])
        app.logger.info('Predicted intent: %s', predictions)
        REQUEST_COUNT.labels('POST', '/intent', '200').inc()
        REQUEST_LATENCY.labels('/intent').observe(time.time() - start_time)
        # Log the successful request and response to the database
        new_log = IntentLog(intent=data['text'], response=str(predictions))
        db.session.add(new_log)
        db.session.commit()
        response_data = jsonify({"intents": predictions})
        # Return the response
        return response_data, 200
    except BadRequest:
        REQUEST_COUNT.labels('POST', '/intent', '400').inc()
        app.logger.error('Invalid JSON format.')
        # Define the format as a dictionary
        format_example = {"text": "Your text here"}
        format_string = json.dumps(format_example)  
        format_string = format_string.replace('\"', '"')
        # Serialize the dictionary to a JSON-formatted string
        return jsonify({"label": "BAD_REQUEST", "message": f"Invalid JSON format."}), 400

@app.errorhandler(500)
def internal_error(error):
    """
    Handle internal server errors.

    :param request:
    :type request:
    :param error:
    :return:
    """
    REQUEST_COUNT.labels(request.method, request.path, '500').inc()
    app.logger.error('Server encountered an internal error: %s', error)
    return jsonify({
        "label": "INTERNAL_ERROR",
        "message": str(error) or "Internal server error."
    }), 500

@app.route('/')
def index():
    """
    ---
    tags:
        - Health

    responses:
        200:
            description: Server is running.
    :param request:
    :return:
    """
    REQUEST_COUNT.labels('GET', '/', '200').inc()
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
