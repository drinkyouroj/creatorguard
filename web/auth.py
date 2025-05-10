from flask_login import LoginManager, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import abort

login_manager = LoginManager()

class User(UserMixin):
    def __init__(self, id, username, email, is_admin=False, is_active=False):
        self.id = id
        self.username = username
        self.email = email
        self.is_admin = is_admin
        self.is_active = is_active

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect('creatorguard.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, is_admin, is_active FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        if user:
            return User(user[0], user[1], user[2], user[3], user[4])
        return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def init_auth_db():
    """Initialize the authentication database tables."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin BOOLEAN DEFAULT 0,
        is_active BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create password reset tokens table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS password_reset_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create initial admin user if none exists
    cursor.execute('SELECT COUNT(*) FROM users WHERE is_admin = 1')
    if cursor.fetchone()[0] == 0:
        register_user(
            'admin',
            'admin@creatorguard.local',
            os.getenv('ADMIN_PASSWORD', 'admin'),
            is_admin=True,
            is_active=True
        )
    
    conn.commit()
    conn.close()

def register_user(username, email, password, is_admin=False, is_active=False):
    """Register a new user."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, is_admin, is_active) VALUES (?, ?, ?, ?, ?)',
            (username, email, generate_password_hash(password), is_admin, is_active)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, password_hash, is_active FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        if not user[2]:  # if not is_active
            return None, "Account not yet activated"
        return User.get(user[0]), None
    return None, "Invalid credentials"

def create_password_reset_token(email):
    """Create a password reset token for a user."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return None
    
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=24)
    
    cursor.execute('DELETE FROM password_reset_tokens WHERE user_id = ?', (user[0],))
    cursor.execute(
        'INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)',
        (user[0], token, expires_at)
    )
    
    conn.commit()
    conn.close()
    return token

def verify_reset_token(token):
    """Verify a password reset token."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, expires_at 
        FROM password_reset_tokens 
        WHERE token = ?
    ''', (token,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return None
        
    user_id, expires_at = result
    expires_at = datetime.fromisoformat(expires_at)
    
    if expires_at < datetime.now():
        cursor.execute('DELETE FROM password_reset_tokens WHERE token = ?', (token,))
        conn.commit()
        conn.close()
        return None
        
    return user_id

def reset_password(token, new_password):
    """Reset a user's password using a valid token."""
    user_id = verify_reset_token(token)
    if not user_id:
        return False
        
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'UPDATE users SET password_hash = ? WHERE id = ?',
        (generate_password_hash(new_password), user_id)
    )
    cursor.execute('DELETE FROM password_reset_tokens WHERE user_id = ?', (user_id,))
    
    conn.commit()
    conn.close()
    return True

def get_pending_users():
    """Get list of users pending activation."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, created_at 
        FROM users 
        WHERE is_active = 0 AND is_admin = 0
        ORDER BY created_at DESC
    ''')
    users = cursor.fetchall()
    conn.close()
    
    return [{
        'id': user[0],
        'username': user[1],
        'email': user[2],
        'created_at': user[3]
    } for user in users]

def get_all_users():
    """Get list of all users (for admin dashboard)."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, is_admin, is_active, created_at 
        FROM users 
        ORDER BY created_at DESC
    ''')
    users = cursor.fetchall()
    conn.close()
    
    return [{
        'id': user[0],
        'username': user[1],
        'email': user[2],
        'is_admin': bool(user[3]),
        'is_active': bool(user[4]),
        'created_at': user[5]
    } for user in users]

def update_user_status(user_id, is_active, is_admin=None):
    """Update a user's active and admin status."""
    conn = sqlite3.connect('creatorguard.db')
    cursor = conn.cursor()
    
    if is_admin is not None:
        cursor.execute(
            'UPDATE users SET is_active = ?, is_admin = ? WHERE id = ?',
            (is_active, is_admin, user_id)
        )
    else:
        cursor.execute(
            'UPDATE users SET is_active = ? WHERE id = ?',
            (is_active, user_id)
        )
    
    conn.commit()
    conn.close()
    return True

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return User.get(user_id)
