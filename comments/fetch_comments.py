import requests
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
DB_PATH = "creatorguard.db"

def fetch_comments(video_id, max_results=100):
    comments = []
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": API_KEY
    }

    while True:
        response = requests.get(base_url, params=params)
        data = response.json()

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "id": item["id"],
                "video_id": video_id,
                "author": snippet.get("authorDisplayName", ""),
                "text": snippet.get("textDisplay", ""),
                "timestamp": snippet.get("publishedAt", "")
            })

        if "nextPageToken" in data and len(comments) < max_results:
            params["pageToken"] = data["nextPageToken"]
        else:
            break

    return comments[:max_results]

def insert_comments(comments):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for comment in comments:
        try:
            cursor.execute("""
                INSERT INTO comments (id, video_id, author, text, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                comment["id"],
                comment["video_id"],
                comment["author"],
                comment["text"],
                comment["timestamp"]
            ))
        except sqlite3.IntegrityError:
            continue  # comment already exists

    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(comments)} comments into the database.")

if __name__ == "__main__":
    video_id = input("🎥 Enter YouTube video ID: ").strip()
    comments = fetch_comments(video_id)
    insert_comments(comments)