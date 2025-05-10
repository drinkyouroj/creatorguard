import os
import sqlite3
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_db():
    """Initialize the database."""
    db_path = 'creatorguard.db'
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    # Backup existing database if it exists
    if os.path.exists(db_path):
        backup_path = f"{db_path}.backup"
        try:
            os.rename(db_path, backup_path)
            print(f"📦 Created backup of existing database at {backup_path}")
        except Exception as e:
            print(f"⚠️ Failed to create backup: {e}")
            return

    # Create new database
    conn = sqlite3.connect(db_path)
    
    try:
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
        
        # Create admin user if it doesn't exist
        admin_password = os.getenv('ADMIN_PASSWORD', 'admin')  # Default password if not set
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO users (username, email, password, is_admin, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'admin',
            'admin@example.com',
            generate_password_hash(admin_password),
            True,
            True
        ))
        
        conn.commit()
        print("✅ Database initialized successfully!")
        print("📝 Schema created and admin user configured")
        
        # Verify the schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("\n📊 Created tables:")
        for table in tables:
            print(f"- {table[0]}")
            # Show columns for each table
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  └─ {col[1]} ({col[2]})")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        # Restore backup if initialization fails
        if os.path.exists(backup_path):
            os.remove(db_path)
            os.rename(backup_path, db_path)
            print("🔄 Restored database from backup")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
