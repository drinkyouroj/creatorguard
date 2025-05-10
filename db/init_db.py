import os
import sqlite3
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def migrate_data(old_conn, new_conn):
    """Migrate data from old database to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Migrate users (handle password to password_hash rename)
        old_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if old_cursor.fetchone():
            old_cursor.execute("SELECT * FROM users")
            users = old_cursor.fetchall()
            
            # Get column names from old table
            old_cursor.execute("PRAGMA table_info(users)")
            columns = old_cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            for user in users:
                # Create a dict of column name to value
                user_data = dict(zip(column_names, user))
                
                # Handle password field rename
                password_value = user_data.get('password_hash', user_data.get('password'))
                
                new_cursor.execute("""
                    INSERT OR IGNORE INTO users 
                    (username, email, password_hash, is_admin, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_data['username'],
                    user_data['email'],
                    password_value,
                    user_data['is_admin'],
                    user_data['is_active']
                ))

        # Migrate comments
        old_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='comments'")
        if old_cursor.fetchone():
            old_cursor.execute("SELECT * FROM comments")
            comments = old_cursor.fetchall()
            
            # Get column names from old table
            old_cursor.execute("PRAGMA table_info(comments)")
            columns = old_cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            for comment in comments:
                comment_data = dict(zip(column_names, comment))
                
                # Insert with new schema fields (they'll be NULL if not in old schema)
                new_cursor.execute("""
                    INSERT OR IGNORE INTO comments 
                    (video_id, comment_id, parent_id, author, text, likes, 
                     reply_count, timestamp, classification, mod_action, 
                     emotional_score, toxicity_score, sentiment_scores, 
                     toxicity_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    comment_data['video_id'],
                    comment_data['comment_id'],
                    comment_data.get('parent_id'),
                    comment_data['author'],
                    comment_data['text'],
                    comment_data.get('likes', 0),
                    comment_data.get('reply_count', 0),
                    comment_data['timestamp'],
                    comment_data.get('classification'),
                    comment_data.get('mod_action'),
                    comment_data.get('emotional_score'),
                    comment_data.get('toxicity_score'),
                    comment_data.get('sentiment_scores'),
                    comment_data.get('toxicity_details')
                ))

        new_conn.commit()
        print("✅ Data migration completed successfully")
        
    except Exception as e:
        print(f"❌ Error during data migration: {e}")
        raise

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
            old_conn = sqlite3.connect(backup_path)
        except Exception as e:
            print(f"⚠️ Failed to create backup: {e}")
            return

    # Create new database
    new_conn = sqlite3.connect(db_path)
    
    try:
        # Create new schema
        with open(schema_path, 'r') as f:
            new_conn.executescript(f.read())
        print("📝 Created new database schema")
        
        # Migrate data if backup exists
        if os.path.exists(backup_path):
            print("🔄 Migrating data from backup...")
            migrate_data(old_conn, new_conn)
        
        # Create admin user if it doesn't exist
        cursor = new_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_password = os.getenv('ADMIN_PASSWORD', 'admin')
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, is_admin, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (
                'admin',
                'admin@example.com',
                generate_password_hash(admin_password),
                True,
                True
            ))
            print("👤 Created admin user")
        
        new_conn.commit()
        print("✅ Database initialization completed successfully!")
        
        # Verify the schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("\n📊 Created tables:")
        for table in tables:
            print(f"- {table[0]}")
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  └─ {col[1]} ({col[2]})")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        # Restore backup if initialization fails
        if os.path.exists(backup_path):
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rename(backup_path, db_path)
            print("🔄 Restored database from backup")
        raise
    finally:
        if 'old_conn' in locals():
            old_conn.close()
        new_conn.close()
        # Keep the backup file for safety
        print(f"💾 A backup of your old database is saved at: {backup_path}")

if __name__ == '__main__':
    init_db()
