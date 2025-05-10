import sqlite3
import os
from werkzeug.security import generate_password_hash

def init_db():
    """Initialize the database with schema and create admin user."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'creatorguard.db')
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

    # Read schema
    with open(schema_path, 'r') as f:
        schema = f.read()

    # Connect to database and create tables
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        
        # Check if admin user exists
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        admin_exists = cursor.fetchone() is not None

        # Create admin user if it doesn't exist
        if not admin_exists:
            admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')  # Default password for development
            cursor.execute(
                'INSERT INTO users (username, email, password, is_admin, is_active) VALUES (?, ?, ?, ?, ?)',
                ('admin', 'admin@example.com', generate_password_hash(admin_password), True, True)
            )
        
        conn.commit()
        print("Database initialized successfully!")
        
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
