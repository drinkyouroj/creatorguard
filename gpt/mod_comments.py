import os
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

class CommentModerator:
    def __init__(self, db_path='creatorguard.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def classify_comments(self, video_id=None, batch_size=10):
        """
        Classify unclassified comments using GPT.
        
        Args:
            video_id (str, optional): Specific video to analyze
            batch_size (int): Number of comments to process in each batch
        """
        # Get unclassified comments
        query = """
            SELECT id, text 
            FROM comments 
            WHERE classification IS NULL
        """
        params = []
        
        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        
        self.cursor.execute(query, params)
        comments = self.cursor.fetchall()
        
        if not comments:
            print("No unclassified comments found.")
            return
        
        print(f" Found {len(comments)} unclassified comments")
        processed = 0
        
        # Process comments in batches
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            
            try:
                # Prepare batch prompt
                comments_text = "\n".join([f"Comment {j+1}: {comment[1]}" for j, comment in enumerate(batch)])
                
                prompt = f"""Analyze these YouTube comments and classify each one:
{comments_text}

For each comment, provide:
1. Classification (positive, negative, neutral, spam, or offensive)
2. Moderation action (allow, flag, or remove)
3. Brief reason for classification

Format each response as:
Comment 1:
- Classification: [classification]
- Action: [action]
- Reason: [brief reason]

Be objective and consistent in your analysis."""

                # Get GPT response
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert content moderator."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                analysis = response.choices[0].message.content
                
                # Parse and update database
                current_comment = 0
                for line in analysis.split('\n'):
                    if line.startswith('Comment '):
                        current_comment = int(line.split()[1].strip(':')) - 1
                    elif line.strip().startswith('- Classification:'):
                        classification = line.split(':')[1].strip().lower()
                    elif line.strip().startswith('- Action:'):
                        action = line.split(':')[1].strip().lower()
                    elif line.strip().startswith('- Reason:'):
                        reason = line.split(':')[1].strip()
                        
                        # Update database
                        if current_comment < len(batch):
                            comment_id = batch[current_comment][0]
                            self.cursor.execute("""
                                UPDATE comments 
                                SET classification = ?, 
                                    mod_action = ?,
                                    reason = ?
                                WHERE id = ?
                            """, (classification, action, reason, comment_id))
                            
                processed += len(batch)
                print(f" Processed {processed}/{len(comments)} comments")
                
                # Commit after each batch
                self.conn.commit()
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        print("\n Comment classification complete!")

    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    try:
        video_id = input(" Enter YouTube video ID (or press Enter for all videos): ").strip()
        video_id = video_id if video_id else None
        
        moderator = CommentModerator()
        moderator.classify_comments(video_id)
        moderator.close()
        
    except Exception as e:
        print(f" Error: {e}")

if __name__ == '__main__':
    main()
