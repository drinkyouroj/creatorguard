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
    login_manager.login_view = 'auth.login'
    
    # Import User model and set up user_loader
    from .auth import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)
    
    # Import and register blueprints
    from . import auth
    from . import views
    
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    
    # Initialize database
    from .auth import init_auth_db
    init_auth_db()
    
    return app
