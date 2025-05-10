import os
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_youtube_comments(video_id, max_results=100):
    """
    Fetch comments for a given YouTube video.
    
    Args:
        video_id (str): YouTube video ID
        max_results (int, optional): Maximum number of comments to fetch. Defaults to 100.
    
    Returns:
        list: A list of comment dictionaries
    """
    # Get YouTube API key from environment
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not youtube_api_key:
        raise ValueError("YouTube API key not found in environment variables")
    
    # Build YouTube API client
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    # Fetch comments
    comments_response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results,
        textFormat='plainText'
    ).execute()
    
    # Process and extract comment details
    comments = []
    for comment_thread in comments_response.get('items', []):
        comment = comment_thread['snippet']['topLevelComment']['snippet']
        comments.append({
            'id': comment_thread['id'],
            'video_id': video_id,
            'author': comment['authorDisplayName'],
            'text': comment['textDisplay'],
            'timestamp': comment['publishedAt']
        })
    
    return comments

def save_comments_to_json(comments, filename='comments.json'):
    """
    Save comments to a JSON file.
    
    Args:
        comments (list): List of comment dictionaries
        filename (str, optional): Output filename. Defaults to 'comments.json'.
    """
    with open(os.path.join('comments', filename), 'w', encoding='utf-8') as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)

def main():
    # Example usage
    video_id = input("🎥 Enter YouTube video ID: ").strip()
    try:
        comments = fetch_youtube_comments(video_id)
        save_comments_to_json(comments)
        print(f"Successfully fetched {len(comments)} comments")
    except Exception as e:
        print(f"Error fetching comments: {e}")

if __name__ == '__main__':
    main()