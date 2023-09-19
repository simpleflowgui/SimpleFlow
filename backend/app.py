# Import the required libraries
from flask import Flask
from flask_cors import CORS


cors = CORS()


def create_app():
    """Application-factory pattern"""
    app = Flask(__name__)
    cors.init_app(app)
    return app
