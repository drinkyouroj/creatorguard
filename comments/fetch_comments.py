import os
import sqlite3
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime

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

    def parse_youtube_timestamp(self, timestamp_str):
        """Convert YouTube timestamp to SQLite timestamp."""
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def fetch_video_metadata(self, video_id):
        """Fetch video metadata from YouTube API."""
        try:
            video_response = self.youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()

            if not video_response.get('items'):
                print(f"❌ Video not found: {video_id}")
                return None

            video = video_response['items'][0]['snippet']
            return {
                'video_id': video_id,
                'title': video.get('title', 'Unknown Title'),
                'channel_title': video.get('channelTitle', 'Unknown Channel'),
                'thumbnail_url': video.get('thumbnails', {}).get('default', {}).get('url')
            }
        except Exception as e:
            print(f"❌ Error fetching video metadata: {e}")
            return None

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
        
        # First, fetch and store video metadata
        video_metadata = self.fetch_video_metadata(video_id)
        if not video_metadata:
            print("❌ Could not fetch video metadata")
            return 0

        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Store video metadata
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO videos 
                (video_id, title, channel_title, thumbnail_url)
                VALUES (?, ?, ?, ?)
            """, (
                video_metadata['video_id'],
                video_metadata['title'],
                video_metadata['channel_title'],
                video_metadata['thumbnail_url']
            ))
            conn.commit()
            print(f"✅ Stored metadata for video: {video_metadata['title']}")
        except sqlite3.Error as e:
            print(f"❌ Error storing video metadata: {e}")
            conn.close()
            return 0
        
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
            conn.close()
            return 0
        
        # Print total comments fetched
        total_fetched = len(comments_response.get('items', []))
        print(f"📥 Total comments fetched from YouTube: {total_fetched}")
        
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
                    (video_id, comment_id, parent_id, author, text, likes, 
                     reply_count, timestamp, classification, mod_action, emotional_score) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    comment_thread['id'],
                    None,  # parent_id is None for top-level comments
                    comment.get('authorDisplayName', 'Unknown'),
                    comment.get('textDisplay', ''),
                    comment.get('likeCount', 0),
                    comment_thread['snippet'].get('totalReplyCount', 0),
                    self.parse_youtube_timestamp(comment.get('publishedAt')),
                    None,  # classification will be set by analysis
                    None,  # mod_action will be set by moderators
                    None   # emotional_score will be set by analysis
                ))
                
                # Track inserted vs duplicate comments
                if cursor.rowcount > 0:
                    inserted_count += 1
                else:
                    duplicate_count += 1
                
                # Fetch and insert replies if they exist
                if comment_thread['snippet']['totalReplyCount'] > 0:
                    replies = self.youtube.comments().list(
                        part='snippet',
                        parentId=comment_thread['id'],
                        maxResults=100  # Adjust as needed
                    ).execute()
                    
                    for reply in replies.get('items', []):
                        reply_snippet = reply['snippet']
                        cursor.execute("""
                            INSERT OR IGNORE INTO comments 
                            (video_id, comment_id, parent_id, author, text, likes, 
                             reply_count, timestamp, classification, mod_action, emotional_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            video_id,
                            reply['id'],
                            comment_thread['id'],
                            reply_snippet.get('authorDisplayName', 'Unknown'),
                            reply_snippet.get('textDisplay', ''),
                            reply_snippet.get('likeCount', 0),
                            0,  # Replies can't have replies
                            self.parse_youtube_timestamp(reply_snippet.get('publishedAt')),
                            None,
                            None,
                            None
                        ))
                        
                        if cursor.rowcount > 0:
                            inserted_count += 1
                        else:
                            duplicate_count += 1
                
            except sqlite3.Error as e:
                print(f"❌ Error inserting comment {comment_thread['id']}: {e}")
                continue
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully processed {total_fetched} comments:")
        print(f"  - {inserted_count} new comments inserted")
        print(f"  - {duplicate_count} duplicate comments skipped")
        
        return inserted_count

def main():
    """Main function for testing."""
    fetcher = YouTubeCommentFetcher()
    video_id = input("Enter YouTube video ID: ")
    fetcher.fetch_comments(video_id)

if __name__ == '__main__':
    main()