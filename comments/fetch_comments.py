import os
import sqlite3
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class YouTubeCommentFetcher:
    def __init__(self, db_path='creatorguard.db'):
        """
        Initialize the comment fetcher with database connection.
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.db_path = db_path
        
        if not self.youtube_api_key:
            raise ValueError("YouTube API key not found in environment variables")
        
        # Build YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)

    def fetch_comments(self, video_id, max_results=100):
        """
        Fetch comments for a given YouTube video and insert into database.
        
        Args:
            video_id (str): YouTube video ID
            max_results (int, optional): Maximum number of comments to fetch
        
        Returns:
            int: Number of comments inserted
        """
        print(f"🔍 Fetching comments for video ID: {video_id}")
        print(f"🔑 Using YouTube API Key: {self.youtube_api_key[:5]}...{self.youtube_api_key[-5:]}")
        
        # Fetch comments from YouTube
        try:
            comments_response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_results,
                textFormat='plainText'
            ).execute()
        except Exception as e:
            print(f"❌ Error fetching comments from YouTube API: {e}")
            return 0
        
        # Print total comments fetched
        total_fetched = len(comments_response.get('items', []))
        print(f"📥 Total comments fetched from YouTube: {total_fetched}")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Counter for inserted comments
        inserted_count = 0
        duplicate_count = 0
        
        # Process and insert comments
        for comment_thread in comments_response.get('items', []):
            comment = comment_thread['snippet']['topLevelComment']['snippet']
            
            try:
                # Insert comment, ignoring duplicates
                cursor.execute("""
                    INSERT OR IGNORE INTO comments 
                    (id, video_id, author, text, timestamp) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    comment_thread['id'],
                    video_id,
                    comment.get('authorDisplayName', 'Unknown'),
                    comment.get('textDisplay', ''),
                    comment.get('publishedAt', '')
                ))
                
                # Track inserted vs duplicate comments
                if cursor.rowcount > 0:
                    inserted_count += 1
                else:
                    duplicate_count += 1
            
            except sqlite3.IntegrityError as e:
                print(f"⚠️ Integrity Error: {e}")
                continue
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        # Detailed output
        print(f"✅ Inserted {inserted_count} new comments")
        print(f"🔁 Skipped {duplicate_count} duplicate comments")
        
        # Verify inserted comments
        self.verify_inserted_comments(video_id)
        
        return inserted_count

    def verify_inserted_comments(self, video_id):
        """
        Verify comments were inserted for the specific video ID.
        
        Args:
            video_id (str): YouTube video ID to verify
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM comments WHERE video_id = ?", (video_id,))
        video_comment_count = cursor.fetchone()[0]
        
        print(f"\n🔍 Verification for video {video_id}:")
        print(f"💬 Total comments in database for this video: {video_comment_count}")
        
        if video_comment_count == 0:
            print("❌ No comments found in database for this video!")
            print("Possible reasons:")
            print("1. Video ID might be incorrect")
            print("2. Comments might be disabled")
            print("3. YouTube API might have restrictions")
        
        conn.close()

def main():
    # Prompt for video ID
    video_id = input("🎥 Enter YouTube video ID: ").strip()
    
    try:
        # Create fetcher and fetch comments
        fetcher = YouTubeCommentFetcher()
        fetcher.fetch_comments(video_id)
    
    except Exception as e:
        print(f"❌ Error fetching comments: {e}")

if __name__ == '__main__':
    main()