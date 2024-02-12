# app/__init__.py

from flask import Flask
from flasgger import Swagger
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True, allow_headers=["Content-Type"], methods=["POST"],
         origins=["http://localhost:3000"])

    # ... (other configurations)

    # Import and register the API blueprint
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Configure Swagger
    Swagger(app)

    return app
