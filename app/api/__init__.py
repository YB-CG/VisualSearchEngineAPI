# app/api/__init__.py

from flask import Blueprint

# Create a Blueprint object for the API routes
api_bp = Blueprint('api', __name__)

# Import the routes module to register the routes
from app.api import routes
