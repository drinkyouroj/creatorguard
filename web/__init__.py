# This file makes the web directory a Python package
import os
from flask import Flask
from flask_login import LoginManager

app = Flask(__name__)
login_manager = LoginManager()

def create_app():
    """Create and configure the Flask application."""
    app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-this')
    
    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    
    # Import and register blueprints
    from . import auth
    from . import views
    
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    
    # Initialize database
    from .auth import init_auth_db
    init_auth_db()
    
    return app
