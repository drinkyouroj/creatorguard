import os
import sqlite3
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()

# Configure OpenAI API
try:
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        http_client=httpx.Client()  # Explicitly create an httpx client
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# Use a valid OpenAI model
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

class CommentAnalyzer:
    def __init__(self, db_path='creatorguard.db'):
        """
        Initialize the comment analyzer with a database connection.
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_comment_stats(self, video_id=None):
        """
        Generate statistical insights about comments.
        
        Args:
            video_id (str, optional): Specific video to analyze. If None, analyze all comments.
        
        Returns:
            dict: Comment statistics
        """
        # Base query with more detailed error checking
        query = "SELECT classification, mod_action, text FROM comments"
        params = []
        
        # Add video_id filter if provided
        if video_id:
            query += " WHERE video_id = ?"
            params.append(video_id)
        
        # Execute query with error handling
        try:
            self.cursor.execute(query, params)
            comments = self.cursor.fetchall()
            
            # Print raw comments for debugging
            print(f"Total comments found: {len(comments)}")
            print("Sample comments:")
            for comment in comments[:5]:
                print(f"Classification: {comment[0]}, Mod Action: {comment[1]}, Text: {comment[2][:50]}...")
            
            # Compute comment statistics
            stats = {
                "total_comments": len(comments),
                "classification_breakdown": dict(Counter(comment[0] for comment in comments if comment[0])),
                "mod_action_breakdown": dict(Counter(comment[1] for comment in comments if comment[1])),
                "sentiment_ratio": {}  # TODO: Implement sentiment ratio calculation
            }
            
            return stats
        
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {}

    def generate_comprehensive_summary(self, stats):
        """
        Generate a comprehensive summary of comment insights using GPT.
        
        Args:
            stats (dict): Comment statistics
        
        Returns:
            str: Comprehensive summary of comment insights
        """
        # Check if client is initialized
        if client is None:
            return "Error: OpenAI client could not be initialized."

        try:
            # Prepare prompt for GPT
            prompt = f"""Analyze the following YouTube comment statistics and provide a comprehensive, insightful summary:

Total Comments: {stats['total_comments']}

Classification Breakdown:
{json.dumps(stats['classification_breakdown'], indent=2)}

Moderation Action Breakdown:
{json.dumps(stats['mod_action_breakdown'], indent=2)}

Provide insights into audience engagement, potential content trends, and recommendations for content creators."""

            # Call OpenAI API 
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful content analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract summary from response
            summary = response.choices[0].message.content
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Unable to generate summary: {str(e)}"

    def save_insights(self, video_id, summary):
        """
        Save comment insights to a markdown file.
        
        Args:
            video_id (str): Video ID for which insights are generated
            summary (str): Comprehensive summary of insights
        """
        # Ensure insights directory exists
        os.makedirs('insights', exist_ok=True)
        
        # Generate filename
        filename = f'insights/comment_insights_{video_id}.md'
        
        # Write insights to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Comment Insights for Video: {video_id}\n\n")
            f.write(summary)
        
        print(f"✅ Insights saved to {filename}")

def main():
    # Prompt for video ID
    video_id = input("🎥 Enter YouTube video ID: ").strip()
    
    try:
        # Create analyzer
        analyzer = CommentAnalyzer()
        
        # Get comment statistics
        stats = analyzer.get_comment_stats(video_id)
        print("Video Comment Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Generate comprehensive summary
        summary = analyzer.generate_comprehensive_summary(stats)
        print("\n--- Comprehensive Summary ---")
        print(summary)
        
        # Save insights
        analyzer.save_insights(video_id, summary)
    
    except Exception as e:
        print(f"❌ Error analyzing comments: {e}")

if __name__ == '__main__':
    main()
