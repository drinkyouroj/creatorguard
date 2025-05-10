import sqlite3
import os

def verify_database(db_path='creatorguard.db'):
    """
    Verify database schema and contents.
    
    Args:
        db_path (str): Path to the SQLite database
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check table schema
    print("🔍 Database Schema:")
    cursor.execute("PRAGMA table_info(comments)")
    columns = cursor.fetchall()
    for column in columns:
        print(f"Column: {column[1]} (Type: {column[2]})")
    
    # Check total number of comments
    cursor.execute("SELECT COUNT(*) FROM comments")
    total_comments = cursor.fetchone()[0]
    print(f"\n📊 Total Comments: {total_comments}")
    
    # Check comments for a specific video
    print("\n🎥 Comments for video 'Ry1IjOft95c':")
    cursor.execute("""
        SELECT id, video_id, author, text, classification, mod_action 
        FROM comments 
        WHERE video_id = ?
        LIMIT 5
    """, ('Ry1IjOft95c',))
    
    video_comments = cursor.fetchall()
    if video_comments:
        for comment in video_comments:
            print(f"ID: {comment[0]}")
            print(f"Video ID: {comment[1]}")
            print(f"Author: {comment[2]}")
            print(f"Text (first 100 chars): {comment[3][:100]}")
            print(f"Classification: {comment[4]}")
            print(f"Mod Action: {comment[5]}")
            print("---")
    else:
        print("No comments found for this video.")
    
    # Close the connection
    conn.close()

def main():
    # Verify the database
    verify_database()

if __name__ == '__main__':
    main()
